import ast
import dataclasses
import importlib.abc
import importlib.util
import re
from functools import reduce
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from bson import ObjectId
from typing_extensions import Literal

from .nirscloud_mongo import MongoMetaBase, RedcapDefnMeta, query_field

_SUBJ_ID_SPLIT = re.compile(r"([_a-zA-Z]+)\s*(\d+)")
_REDCAP_NUM_REGEX = re.compile(r"[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?")
NAN = float("nan")
REDCAP_TEXT_TO_NUM_FIXES = {
    "": NAN,
    ".13.2": 13.2,
    "7,2": 7.2,
    "4.;0": 4.0,
}


class RedcapEntryMeta(MongoMetaBase, database_name="cchu_redcap2"):
    mongo: ObjectId = query_field("_id")
    meta: str = query_field("meta_id")
    subject_id: str = query_field("Subject ID")
    subject: int = dataclasses.field(init=False)
    data_enterer: str = dataclasses.field(init=False)
    study_id: Optional[str] = query_field("study-id", default=None)

    def __post_init__(self):
        self.data_enterer, sid = _SUBJ_ID_SPLIT.match(self.subject_id).groups()
        self.subject = int(sid)


def _convert_redcap_yesno(value: Literal["Yes", "No"]) -> Optional[bool]:
    if value == "Yes":
        return True
    elif value == "No":
        return False
    elif value == "":
        return None
    else:
        return None
        # raise ValueError(value)


def _convert_redcap_no_empty(value: str) -> Optional[str]:
    return None if value == "" else value


def _convert_redcap_calc(value: str) -> Optional[float]:
    return None if value == "" else (float(value) if "." in value else int(value))


def _convert_redcap_text_num(value: str) -> float:
    global REDCAP_TEXT_TO_NUM_FIXES
    if value in REDCAP_TEXT_TO_NUM_FIXES:
        return REDCAP_TEXT_TO_NUM_FIXES[value]
    matches = [m.group() for m in _REDCAP_NUM_REGEX.finditer(value)]
    if len(matches) > 1:
        raise ValueError(f"More than one number found in {value!r}")
    elif len(matches) == 1:
        return float(matches[0])
    else:
        return NAN


def _convert_redcap_text_date(value: str) -> np.datetime64:
    try:
        return np.datetime64(value, "ns")
    except ValueError:
        return np.datetime64("NaT", "ns")


def _convert_redcap_text_time(value: str) -> np.timedelta64:
    if value == "":
        return np.timedelta64("NaT", "ns")
    else:
        hh, mm = value.split(":")
        return (np.timedelta64(hh, "h") + np.timedelta64(mm, "m")).astype("timedelta64[ns]")


def _gen_redcap_imports():
    return [
        ast.ImportFrom("typing", [ast.alias("Optional")], level=0),
        ast.ImportFrom("typing_extensions", [ast.alias("Literal")], level=0),
        ast.ImportFrom("numpy", [ast.alias("datetime64"), ast.alias("timedelta64")], level=0),
        ast.ImportFrom(
            None,
            [ast.alias("RedcapEntryMeta"), ast.alias("query_field"), ast.alias("REDCAP_TEXT_TO_NUM_FIXES")],
            level=1,
        ),
    ]


def _gen_redcap_cls_body(form_fields: "list[RedcapDefnMeta]") -> Iterator[ast.AnnAssign]:
    for f in form_fields:
        kwargs = {}
        if f.ty in ("text", "notes", "descriptive"):
            if f.text_validation_type_or_show_slider_number == "number":
                ty = ast.Name("float", ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_text_num", ctx=ast.Load())
            elif f.text_validation_type_or_show_slider_number == "date_dmy":
                ty = ast.Name("datetime64", ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_text_date", ctx=ast.Load())
            elif f.text_validation_type_or_show_slider_number == "time":
                ty = ast.Name("timedelta64", ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_text_time", ctx=ast.Load())
            elif f.text_validation_type_or_show_slider_number is None:
                ty = ast.Name("str", ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_no_empty", ctx=ast.Load())
            else:
                raise NotImplementedError(f.text_validation_type_or_show_slider_number, f)
        elif f.ty in ("radio", "dropdown", "checkbox"):
            choices = f.select_choices_or_calculations.split(" | ")
            choices = dict(c.split(", ", maxsplit=1) for c in choices)
            if f.ty == "checkbox":
                # skip for now
                continue
            else:
                ty = list(map(ast.Constant, choices.values()))
                ty = ast.Subscript(ast.Name("Literal", ctx=ast.Load()), ast.Tuple(ty, ctx=ast.Load()), ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_no_empty", ctx=ast.Load())
        elif f.ty == "yesno":
            ty = ast.Name("bool", ctx=ast.Load())
            kwargs["converter"] = ast.Name("_convert_redcap_yesno", ctx=ast.Load())
        elif f.ty == "calc":
            ty = ast.Name("float", ctx=ast.Load())
            kwargs["converter"] = ast.Name("_convert_redcap_calc", ctx=ast.Load())
        else:
            raise NotImplementedError(f.ty, f)
        if f.required_field != "y":
            ty = ast.Subscript(ast.Name("Optional", ctx=ast.Load()), ty, ctx=ast.Load())
            kwargs["default"] = ast.Constant(None)
        yield ast.AnnAssign(
            target=ast.Name(f.name, ctx=ast.Store()),
            annotation=ty,
            simple=True,
            value=ast.Call(
                func=ast.Name("query_field", ctx=ast.Load()),
                args=[ast.Constant(f.name)],
                keywords=[ast.keyword(arg=k, value=v) for k, v in kwargs.items()],
            ),
        )


class _AstLoader(importlib.abc.InspectLoader):
    def __init__(self, module_ast: ast.Module):
        self.module_ast = module_ast

    def get_source(self, fullname):
        return ast.unparse(self.module_ast)


def create_redcap_cls_module(form_field_defns: "list[RedcapDefnMeta]", form_cls_map):
    form_defns = reduce(lambda grp, f: grp.setdefault(f.form, []).append(f) or grp, form_field_defns, {})
    redcap_clss = [
        ast.ClassDef(
            name=cls_name,
            bases=[ast.Name("RedcapEntryMeta", ctx=ast.Load())],
            keywords=[
                ast.keyword(arg="database_name", value=ast.Constant("cchu_redcap2")),
                ast.keyword(
                    arg="default_query",
                    value=ast.Dict([ast.Constant(f"{form_name}_complete.val")], [ast.Constant("Complete")]),
                ),
            ],
            body=list(_gen_redcap_cls_body(form_defns[form_name])),
            decorator_list=[],
        )
        for form_name, cls_name in form_cls_map.items()
    ]
    redcap_clss_mod = ast.fix_missing_locations(
        ast.Module(body=[*_gen_redcap_imports(), *redcap_clss], type_ignores=[])
    )
    # runtime generated code is supposed to use filenames in the form of "<something>"
    unique_fpath = f"<generated from 0x{id(form_field_defns):x}>"
    # FIXME: `linecache` doesn't allow filenames in the form of "<something>", `doctest` monkeypatches `linecache` module to enable inspection.
    unique_fpath = str(Path(__file__).parent / unique_fpath)
    spec = importlib.util.spec_from_loader(
        f"{__name__}.redcap", loader=_AstLoader(redcap_clss_mod), origin=unique_fpath
    )
    spec.has_location = True
    redcap = importlib.util.module_from_spec(spec)
    redcap.__dict__.update(
        _convert_redcap_yesno=_convert_redcap_yesno,
        _convert_redcap_no_empty=_convert_redcap_no_empty,
        _convert_redcap_calc=_convert_redcap_calc,
        _convert_redcap_text_num=_convert_redcap_text_num,
        _convert_redcap_text_date=_convert_redcap_text_date,
        _convert_redcap_text_time=_convert_redcap_text_time,
    )
    spec.loader.exec_module(redcap)
    return redcap
