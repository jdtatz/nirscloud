import ast
import dataclasses
import importlib.util
import re
from functools import reduce
from typing import Iterator, Optional

import numpy as np
from bson import ObjectId
from typing_extensions import Literal

from .nirscloud_mongo import MongoMetaBase, RedcapDefnMeta, query_field

_SUBJ_ID_SPLIT = re.compile(r"([_a-zA-Z]+)\s*(\d+)")
_REDCAP_NUM_REGEX = re.compile(r"[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?")
NAN = float("nan")


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
    if value == "":
        return NAN
    elif value == ".13.2":
        return 13.2
    elif value == "7,2":
        return 7.2
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


def _gen_redcap_cls_body(form_fields: "list[RedcapDefnMeta]") -> Iterator[ast.AnnAssign]:
    for f in form_fields:
        kwargs = {}
        if f.ty in ("text", "notes", "descriptive"):
            if f.text_validation_type_or_show_slider_number == "number":
                ty = ast.Name("float", ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_text_num", ctx=ast.Load())
            elif f.text_validation_type_or_show_slider_number == "date_dmy":
                ty = ast.Attribute(ast.Name("np", ctx=ast.Load()), "datetime64", ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_text_date", ctx=ast.Load())
            elif f.text_validation_type_or_show_slider_number == "time":
                ty = ast.Attribute(ast.Name("np", ctx=ast.Load()), "timedelta64", ctx=ast.Load())
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


def create_redcap_cls_module(form_field_defns: "list[RedcapDefnMeta]", form_cls_map, *, return_src: bool = False):
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
    redcap_clss_mod = ast.Module(body=redcap_clss, type_ignores=[])
    code = compile(ast.fix_missing_locations(redcap_clss_mod), "<string>", "exec")
    spec = importlib.util.spec_from_loader("redcap", loader=None)
    redcap = importlib.util.module_from_spec(spec)
    exec(code, None, redcap.__dict__)
    if return_src:
        return redcap, ast.unparse(redcap_clss_mod)
    else:
        return redcap
