import ast
import importlib.abc
import importlib.util
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, ClassVar, Iterator, Literal, Optional, Self, TypedDict, TypeVar

import numpy as np
from redcap import Project
from redcap.methods.metadata import Metadata
from redcap.methods.records import Records
from typing_extensions import dataclass_transform

_SUBJ_ID_SPLIT = re.compile(r"([_a-zA-Z]+)\s*(\d+)")
_REDCAP_NUM_REGEX = re.compile(r"[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?")
NAN = float("nan")
REDCAP_TEXT_TO_NUM_FIXES = {
    "": NAN,
    ".13.2": 13.2,
    "7,2": 7.2,
    "4.;0": 4.0,
}

JSONValue = Mapping[str, "JSONValue"] | Sequence["JSONValue"] | str | int | float | bool | None


class TypedRedcapMetadata(TypedDict):
    field_name: str
    form_name: str
    section_header: str
    field_type: Literal["dropdown", "notes", "text", "yesno", "descriptive", "calc", "radio", "checkbox"]
    field_label: str
    select_choices_or_calculations: str
    field_note: str
    text_validation_type_or_show_slider_number: Literal["", "time", "date_dmy", "number"]
    text_validation_min: str
    text_validation_max: str
    identifier: str
    branching_logic: str
    required_field: Literal["", "y", "n"]
    custom_alignment: Literal["", "LV", "LH", "RH"]
    question_number: str
    matrix_group_name: str
    matrix_ranking: str
    field_annotation: str


_T = TypeVar("_T")


def _id_cast(v: JSONValue) -> _T:
    return v


def redcap_field(
    key: str,
    converter: Callable[[JSONValue], _T] = _id_cast,
    default: _T | Literal[MISSING] = MISSING,
    default_factory: Callable[[], _T] | Literal[MISSING] = MISSING,
    **field_kwargs,
):
    default_factory = (lambda: default) if default is not MISSING else default_factory
    return field(
        metadata=dict(
            redcap_key=key,
            redcap_converter=converter,
            redcap_default_factory=default_factory,
        ),
        **field_kwargs,
    )


@dataclass_transform(field_specifiers=(redcap_field,), kw_only_default=True)
class TypedRedcapRecordMetaBase:
    _form_name: ClassVar[str]
    _record_id_name: ClassVar[str]

    def __init_subclass__(
        cls,
        form_name: str,
        record_id_name: str,
        *args,
        init: bool = True,
        frozen: bool = False,
        eq: bool = True,
        order: bool = True,
        **kwargs,
    ):
        cls._form_name = form_name
        cls._record_id_name = record_id_name
        super().__init_subclass__(*args, **kwargs)
        dataclass(cls, init=init, frozen=frozen, eq=eq, order=order)

    @classmethod
    def _redcap_keys(cls):
        yield from (f.metadata.get("redcap_key", f.name) for f in fields(cls))

    @classmethod
    def _redcap_defaults(cls):
        for f in fields(cls):
            default_factory = f.metadata.get("redcap_default_factory", MISSING)
            if default_factory is not MISSING:
                yield (f.name, default_factory())

    @classmethod
    def _redcap_converters(cls) -> dict[str, tuple[str, Callable[[JSONValue], Any]]]:
        return {
            f.metadata.get("redcap_key", f.name): (
                f.name,
                f.metadata.get("redcap_converter", _id_cast),
            )
            for f in fields(cls)
        }

    @classmethod
    def from_record(cls, record: dict[str, JSONValue]) -> Self:
        converters = cls._redcap_converters()
        rdefaults = dict(cls._redcap_defaults())
        rfields = dict()
        extra = dict()
        for rk, rv in record.items():
            converter = converters.get(rk, None)
            if converter is None:
                extra[rk] = rv
            else:
                k, f = converter
                rfields[k] = f(rv)
        obj = cls(**{**rdefaults, **rfields})
        obj._extra = extra
        return obj

    @classmethod
    def export_records(
        cls,
        client: Records,
        filter_logic: Optional[str] = None,
        date_begin: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
    ):
        form_filter_logic = f"[{cls._form_name}_complete] = 2"
        if filter_logic is None:
            filter_logic = form_filter_logic
        else:
            filter_logic = f"{form_filter_logic} AND ({filter_logic})"
        # NOTE: Setting `export_checkbox_labels=True` is only an issue if an empty string is a valid checkbox label
        for record in client.export_records(
            forms=[cls._form_name],
            filter_logic=filter_logic,
            date_begin=date_begin,
            date_end=date_end,
            raw_or_label="label",
            export_checkbox_labels=True,
        ):
            yield cls.from_record(record)


class TypedRedcapRecordBase(TypedRedcapRecordMetaBase, form_name="", record_id_name=""):
    subject: int = field(init=False)
    data_enterer: str = field(init=False)

    def __post_init__(self):
        self.data_enterer, sid = _SUBJ_ID_SPLIT.match(getattr(self, self._record_id_name)).groups()
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
            [
                ast.alias("TypedRedcapRecordMetaBase"),
                ast.alias("TypedRedcapRecordBase"),
                ast.alias("redcap_field"),
                ast.alias("REDCAP_TEXT_TO_NUM_FIXES"),
            ],
            level=1,
        ),
    ]


def _gen_redcap_cls_body(*form_fields: TypedRedcapMetadata) -> Iterator[ast.AnnAssign]:
    for f in form_fields:
        kwargs = {}
        if f["field_type"] in ("text", "notes", "descriptive"):
            kind = f["text_validation_type_or_show_slider_number"]
            if kind == "number":
                ty = ast.Name("float", ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_text_num", ctx=ast.Load())
            elif kind == "date_dmy":
                ty = ast.Name("datetime64", ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_text_date", ctx=ast.Load())
            elif kind == "time":
                ty = ast.Name("timedelta64", ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_text_time", ctx=ast.Load())
            elif kind == "" or kind is None:
                ty = ast.Name("str", ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_no_empty", ctx=ast.Load())
            else:
                raise NotImplementedError(kind, f)
        elif f["field_type"] in ("radio", "dropdown", "checkbox"):
            choices = f["select_choices_or_calculations"].split(" | ")
            choices = dict(c.split(", ", maxsplit=1) for c in choices)
            if f["field_type"] == "checkbox":
                # skip for now
                continue
            else:
                ty = list(map(ast.Constant, choices.values()))
                ty = ast.Subscript(ast.Name("Literal", ctx=ast.Load()), ast.Tuple(ty, ctx=ast.Load()), ctx=ast.Load())
                kwargs["converter"] = ast.Name("_convert_redcap_no_empty", ctx=ast.Load())
        elif f["field_type"] == "yesno":
            ty = ast.Name("bool", ctx=ast.Load())
            kwargs["converter"] = ast.Name("_convert_redcap_yesno", ctx=ast.Load())
        elif f["field_type"] == "calc":
            ty = ast.Name("float", ctx=ast.Load())
            kwargs["converter"] = ast.Name("_convert_redcap_calc", ctx=ast.Load())
        else:
            raise NotImplementedError(f["field_type"], f)
        if f["required_field"] != "y":
            ty = ast.Subscript(ast.Name("Optional", ctx=ast.Load()), ty, ctx=ast.Load())
            kwargs["default"] = ast.Constant(None)
        yield ast.AnnAssign(
            target=ast.Name(f["field_name"], ctx=ast.Store()),
            annotation=ty,
            simple=True,
            value=ast.Call(
                func=ast.Name("redcap_field", ctx=ast.Load()),
                args=[ast.Constant(f["field_name"])],
                keywords=[ast.keyword(arg=k, value=v) for k, v in kwargs.items()],
            ),
        )


class _AstLoader(importlib.abc.InspectLoader):
    def __init__(self, module_ast: ast.Module):
        self.module_ast = module_ast

    def get_source(self, fullname):
        return ast.unparse(self.module_ast)

    def get_code(self, fullname):
        return compile(self.module_ast, "<string>", "exec", dont_inherit=True)


# TODO: Handle `redcap_event_name`, `redcap_repeat_instrument`, and `redcap_repeat_instance`
def create_typed_redcap_cls_module(project: Project, form_cls_map: dict[str, str]):
    # runtime generated code is supposed to use filenames in the form of "<something>"
    unique_fpath = f"<generated from 0x{id(project):x}>"
    form_field_defns: list[TypedRedcapMetadata] = list(project.metadata)
    (record_id_idx,) = (i for i, m in enumerate(form_field_defns) if m["field_name"] == project.def_field)
    record_id_defn = form_field_defns.pop(record_id_idx)
    form_defns = reduce(lambda grp, f: grp.setdefault(f["form_name"], []).append(f) or grp, form_field_defns, {})
    redcap_clss = [
        ast.ClassDef(
            name=cls_name,
            bases=[ast.Name("TypedRedcapRecordBase", ctx=ast.Load())],
            keywords=[
                ast.keyword(arg="form_name", value=ast.Constant(form_name)),
                ast.keyword(arg="record_id_name", value=ast.Constant(project.def_field)),
            ],
            body=list(_gen_redcap_cls_body(record_id_defn, *form_defns[form_name])),
            decorator_list=[],
        )
        for form_name, cls_name in form_cls_map.items()
    ]
    redcap_clss_mod = ast.fix_missing_locations(
        ast.Module(body=[*_gen_redcap_imports(), *redcap_clss], type_ignores=[])
    )
    # # FIXME: `linecache` doesn't allow filenames in the form of "<something>", `doctest` monkeypatches `linecache` module to enable inspection.
    # unique_fpath = str(Path(__file__).parent / unique_fpath)
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


import functools
import inspect

_getsource = inspect.getsource


@functools.wraps(_getsource)
def getsource(m):
    """`inspect.getsource` doesn't use the fast path for modules and can fail for generated modules"""
    if inspect.ismodule(m):
        spec = m.__spec__
        if spec.loader and isinstance(spec.loader, importlib.abc.InspectLoader):
            src = spec.loader.get_source(spec.name)
            if src:
                return src
    return _getsource(m)


def monkeypatch_inspect_getsource():
    global _monkeypatch_applied
    if not _monkeypatch_applied:
        inspect.getsource = getsource
    _monkeypatch_applied = True


_monkeypatch_applied = False
