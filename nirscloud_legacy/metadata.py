import warnings
from typing import NamedTuple

import numpy as np
import xarray as xr

from .mongo import DCSMeta, FastrakMeta, FinapresMeta, Meta, MetaID, MetaOxMeta, NIRSMeta

type Attrs = dict[str, str | MetaID]


class Metadata(NamedTuple):
    coords: xr.Coordinates
    attrs: Attrs
    var_attrs: dict[str, Attrs]


def _safe_update_attrs(ds: xr.Variable | xr.DataArray | xr.Dataset, attrs: Attrs):
    for k, v in attrs.items():
        if k in ds.attrs and ds.attrs[k] != v:
            msg = f"overriding attribute {k!r} from {ds.attrs[k]!r} to {v!r}"
            warnings.warn(msg, stacklevel=2)
    ds.attrs.update(attrs)


def update_metadata(ds: xr.Dataset, metadata: Metadata) -> xr.Dataset:
    ds.coords.update(metadata.coords)
    _safe_update_attrs(ds, metadata.attrs)
    for k, attrs in metadata.var_attrs.items():
        if k in ds.data_vars:
            _safe_update_attrs(ds[k], attrs)
    return ds


def try_pop_extra_attrs(ds: xr.DataArray | xr.Dataset, time_coord: str = "time"):
    if time_coord not in ds.coords:
        return ds
    time = ds.coords[time_coord]
    if not np.isdtype(time.dtype, np.datetime64):
        return ds
    if "start" in ds.attrs:
        start = ds.attrs["start"]
        startv = time[0].values
        if start != startv:
            msg = f"start times differ between the metadata {start!r} and the data {startv}"
            warnings.warn(msg, stacklevel=2)
        else:
            ds.attrs.pop("start")
    if "end" in ds.attrs:
        end = ds.attrs["end"]
        endv = time[-1].values
        if end != endv:
            msg = f"end times differ between the metadata {end!r} and the data {endv}"
            warnings.warn(msg, stacklevel=2)
        else:
            ds.attrs.pop("end")
    return ds


def _to_datetime_scalar(v, unit="D"):
    if isinstance(v, np.generic):
        return v.astype(f"datetime64[{unit}]")
    else:
        return np.datetime64(v, unit)


def _convert_base_meta(meta: Meta) -> Attrs:
    attrs = {
        "subject": meta.subject,
        "date": _to_datetime_scalar(meta.date),
        "measurement": meta.measurement,
        "meta_id": meta.meta,
    }
    if meta.study:
        attrs["study"] = meta.study
    if meta.group:
        attrs["group"] = meta.group
    if meta.postfix:
        attrs["postfix"] = meta.postfix
    if meta.operators:
        attrs["operators"] = meta.operators
    if meta.session:
        attrs["session"] = meta.session
    if meta.device:
        attrs["device"] = meta.device
    if meta.note_meta:
        attrs["note_id"] = meta.note_meta
    if meta.measurement_notes:
        attrs["notes"] = meta.measurement_notes
    return attrs


def convert_nirs_meta(meta: NIRSMeta | MetaOxMeta) -> Metadata:
    attrs = _convert_base_meta(meta)

    if meta.duration is not None:
        attrs["duration"] = meta.duration
    if meta.nirs_start is not None:
        attrs["start"] = meta.nirs_start
    if meta.nirs_end is not None:
        attrs["end"] = meta.nirs_end
    if meta.nirs_hz is not None:
        attrs["nirs_hz"] = meta.nirs_hz

    coords = {}
    coords["rho"] = "detector", np.array(meta.nirs_distances), {"units": "cm"}
    if meta.nirs_wavelengths is not None:
        coords["wavelength"] = "wavelength", np.array(meta.nirs_wavelengths), {"units": "nm"}
    if meta.gains is not None:
        coords["gain"] = "detector", np.array(meta.gains)
    coords = xr.Coordinates(coords)

    var_attrs = {"phase": {"units": "radian" if meta.is_radian else "degree"}}

    return Metadata(coords, attrs, var_attrs)


def convert_dcs_meta(meta: DCSMeta | MetaOxMeta) -> Metadata:
    attrs = _convert_base_meta(meta)

    if meta.duration is not None:
        attrs["duration"] = meta.duration
    if meta.dcs_start is not None:
        attrs["start"] = meta.dcs_start
    if meta.dcs_end is not None:
        attrs["end"] = meta.dcs_end
    if meta.dcs_hz is not None:
        attrs["dcs_hz"] = meta.dcs_hz

    coords = {}
    coords["rho"] = "channel", np.array(meta.dcs_distances), {"units": "cm"}
    if meta.dcs_wavelength is not None:
        coords["wavelength"] = (), np.array(meta.dcs_wavelength), {"units": "nm"}
    coords = xr.Coordinates(coords)

    var_attrs = {}

    return Metadata(coords, attrs, var_attrs)


def convert_fastrak_meta(meta: FastrakMeta) -> Metadata:
    attrs = _convert_base_meta(meta)

    coords = xr.Coordinates()

    # NOTE: support both the raw and converted fastrak datasets
    pos_attrs = {"units": "cm" if meta.is_cm else "inch"}
    ang_attrs = {"units": "degree"}
    var_attrs = {
        "position": pos_attrs,
        "x": pos_attrs,
        "y": pos_attrs,
        "z": pos_attrs,
        "a": ang_attrs,
        "e": ang_attrs,
        "r": ang_attrs,
    }

    return Metadata(coords, attrs, var_attrs)


def convert_finapres_meta(meta: FinapresMeta) -> Metadata:
    attrs = _convert_base_meta(meta)
    coords = xr.Coordinates()
    # TODO: add timezone info
    var_attrs = {}
    return Metadata(coords, attrs, var_attrs)
