from pathlib import PosixPath
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from .nirscloud_mongo import (
    DCSMeta,
    FastrakMeta,
    FinapresMeta,
    Meta,
    NIRSMeta,
    PatientMonitorMeta,
)


def read_nirsraw(nirsraw_path: PosixPath, nirsraw_rhos: tuple, nirsraw_wavelengths: tuple):
    if nirsraw_path.parts[:2] == ("/", "smb"):
        nirsraw_path = PosixPath("/", "home", *nirsraw_path.parts[2:])
    with nirsraw_path.open() as f:
        nirsraw = np.genfromtxt(f, delimiter="\t")
    ndet = len(nirsraw_rhos)
    nwave = len(nirsraw_wavelengths)
    _idx, raw_data, aux, dark, _empty = np.split(
        nirsraw, np.add.accumulate([1] + [nwave * 3 * ndet] + [ndet] * 2), axis=1
    )
    assert np.allclose(np.diff(_idx[:, 0]), 1) and _idx[0, 0] == 0
    assert _empty.size == 0
    stacked_idx = pd.MultiIndex.from_product(
        (nirsraw_rhos, nirsraw_wavelengths, ("ac", "dc", "phase")), names=["rho", "wavelength", "variable"]
    )
    stacked_data = xr.DataArray(
        raw_data,
        dims=("time", "stacked_rho_wavelength_variable"),
        coords={
            "stacked_rho_wavelength_variable": stacked_idx,
        },
    )
    return (
        stacked_data.to_unstacked_dataset("stacked_rho_wavelength_variable", 2)
        .unstack("stacked_rho_wavelength_variable")
        .assign(
            aux=(("time", "rho"), aux),
            dark=(("time", "rho"), dark),
        )
    )


def _to_datetime_scalar(v, unit="D"):
    if isinstance(v, np.generic):
        return v.astype(f"datetime64[{unit}]")
    else:
        return np.datetime64(v, unit)


# polyfill for <3.9
if hasattr(str, "removeprefix"):
    str_removeprefix = str.removeprefix
else:

    def _removeprefix(self: str, prefix: str) -> str:
        if self.startswith(prefix):
            return self[len(prefix) :]
        else:
            return self[:]

    str_removeprefix = _removeprefix


def _maybe_rsplit_once(s: str, sep: Optional[str]):
    head, *tail = s.rsplit(sep, 1)
    if len(tail) == 0:
        return head, None
    elif len(tail) == 1:
        (rest,) = tail
        return head, rest
    raise RuntimeError(
        f"builtin `str.rsplit(sep, maxsplit=1)` broke invariants. Should've only returned 2 values at most, but returned {1 + len(tail)} values"
    )


def add_meta_coords(ds: xr.Dataset, meta: Meta):
    loc, trial = _maybe_rsplit_once(meta.measurement, "_")
    trial = None if trial is None else int(trial)
    session = int(str_removeprefix(meta.session, "S")) if meta.session else None

    coords = {
        "study": meta.study,
        # TODO: split into (`group`, `subject_id`) once ensured all instances of this field follow this pattern
        "subject": meta.subject,
        "session": session,
        "device": meta.device,
        # FIXME: deprecate later (`location`, `trial`) == `measurement`
        "measurement": meta.measurement,
        "location": loc,
        "trial": trial,
        "date": _to_datetime_scalar(meta.date),
        "note_id": meta.note_meta,
        "meta_id": meta.meta,
        # "group": meta.group,
    }
    if isinstance(meta, NIRSMeta):
        start, end = ds["time"][0], ds["time"][-1]
        dt_dur = np.round((end - start) / np.timedelta64(1, "s")).astype("timedelta64[s]")
        coords.update(
            nirs_start_time=start,
            nirs_end_time=end,
            duration=meta.duration if meta.duration is not None else dt_dur,
            rho=("rho", np.array(meta.nirs_distances), dict(units="cm")),
            frequency=((), meta.nirs_hz, dict(units="Hz")),
        )
        if meta.nirs_wavelengths is not None:
            coords["wavelength"] = "wavelength", np.array(meta.nirs_wavelengths), dict(units="nm")
        if meta.gains is not None:
            coords["gain"] = "rho", np.array(meta.gains)
        ds["time"] = ds["time"] - ds["time"][0]
        ds = ds.assign(phase=ds["phase"].assign_attrs(units="radian" if meta.is_radian else "deg"))
    elif isinstance(meta, DCSMeta):
        start, end = ds["time"][0], ds["time"][-1]
        dt_dur = np.round((end - start) / np.timedelta64(1, "s")).astype("timedelta64[s]")
        coords.update(
            dcs_start_time=start,
            dcs_end_time=end,
            duration=meta.duration if meta.duration is not None else dt_dur,
            rho=("channel", np.array(meta.dcs_distances), dict(units="cm")),
        )
        if meta.dcs_hz is not None:
            coords["frequency"] = (), np.array(meta.dcs_hz), dict(units="Hz")
        if meta.dcs_wavelength is not None:
            coords["wavelength"] = (), np.array(meta.dcs_wavelength), dict(units="nm")
        ds["time"] = ds["time"] - ds["time"][0]
    elif isinstance(meta, FastrakMeta):
        position = ds["position"]
        if not meta.is_cm:
            position = position * 2.54
        ds = ds.assign(position=position.assign_attrs(units="cm"))
    elif isinstance(meta, FinapresMeta):
        pass
    elif isinstance(meta, PatientMonitorMeta):
        pass
    else:
        # raise TypeError
        pass
    return ds.assign_coords(coords)


def _chunked_shape(array: np.ndarray):
    if array.dtype.kind != "O":
        return array.shape
    return (*array.shape, *_chunked_shape(array[0]))


def _chunked_conc(array: np.ndarray):
    if array.dtype.kind != "O":
        return array
    return _chunked_conc(np.concatenate(array, axis=None))


def _from_chunked_array(carray: pa.ChunkedArray) -> np.ndarray:
    array = carray.to_numpy()
    if not pa.types.is_nested(carray.type):
        return array
    shape = _chunked_shape(array)
    return _chunked_conc(array).reshape(shape)


def _unique_squeezed_attr_or_drop(da: xr.DataArray):
    ks = [k for k in da.coords if k not in da.dims]
    for k in ks:
        v = np.unique(da[k]).squeeze()
        da = da.drop_vars(k)
        if v.ndim == 0:
            da = da.assign_attrs({k: v})
    return da


def _time_from_table(table: pa.Table):
    if "_nano_ts" in table.column_names:
        return _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]")
    elif "_milli_ts" in table.column_names:
        return _from_chunked_array(table["_milli_ts"]).astype("datetime64[ms]")
    else:
        raise KeyError(f"No known time column, `_nano_ts` or `_milli_ts`, found in the table. [{table.column_names}]")


def nirs_ds_from_table(table: pa.Table):
    return (
        xr.Dataset(
            data_vars=dict(
                ac=(("time", "wavelength", "rho"), _from_chunked_array(table["ac"])),
                phase=(("time", "wavelength", "rho"), _from_chunked_array(table["phase"])),
                dc=(("time", "wavelength", "rho"), _from_chunked_array(table["dc"])),
                dark=(("time", "rho"), _from_chunked_array(table["dark"])),
                aux=(("time", "rho"), _from_chunked_array(table["aux"])),
            ),
            coords=dict(time=("time", _time_from_table(table))),
        )
        .transpose(..., "time")
        .sortby("time")
    )


def dcs_ds_from_table(table: pa.Table):
    tau = _from_chunked_array(table["t"])
    # assert np.unique(tau, axis=0).shape[0] == 1
    return (
        xr.Dataset(
            data_vars=dict(
                counts=(("time", "channel"), _from_chunked_array(table["CPS"]), dict(units="Hz")),
                # t=(("time", "tau"), tau),
                value=(("time", "tau", "channel"), _from_chunked_array(table["t_val"])),
            ),
            coords=dict(
                time=("time", _time_from_table(table)),
                tau=("tau", tau[0]),
            ),
        )
        .transpose(..., "time")
        .sortby("time")
    )


def fastrak_ds_from_table(table: pa.Table):
    # idx=0 is always the pen, idx=1 is always the nirs sensor, and if idx=2 exists then it's the head sensor (refrence point for dual-quat transformation)
    from scipy.spatial.transform import Rotation

    cartesian_axes = "x", "y", "z"
    euler_axes = "a", "e", "r"
    quaternion_axes = "w", "x", "y", "z"

    position = np.stack([_from_chunked_array(table[c]) for c in cartesian_axes], axis=0)
    angles = np.stack([_from_chunked_array(table[c]) for c in euler_axes], axis=1)
    orientation = np.roll(Rotation.from_euler("ZYX", angles, degrees=True).as_quat().transpose(), 1, axis=0)
    ds = xr.Dataset(
        data_vars=dict(
            position=(("cartesian_axes", "stacked"), position),
            orientation=(("quaternion_axes", "stacked"), orientation),
        ),
        coords=dict(
            idx=("stacked", _from_chunked_array(table["idx"])),
            time=("stacked", _time_from_table(table)),
            cartesian_axes=("cartesian_axes", list(cartesian_axes)),
            quaternion_axes=("quaternion_axes", list(quaternion_axes)),
        ),
    )
    ds = ds.set_index(stacked=["time", "idx"]).unstack("stacked").transpose("idx", ..., "time")
    return ds.sortby("time")


def finapres_ds_from_table(table: pa.Table):
    return xr.Dataset(
        data_vars=dict(
            pressure=("time", _from_chunked_array(table["pressure_mmHg"]), dict(units="mmHg")),
            # height=("time", _from_chunked_array(table["height_mmHg"]), dict(units="mmHg")),
            plethysmograph=("time", _from_chunked_array(table["plethysmograph"])),
        ),
        coords=dict(
            time=("time", _time_from_table(table)),
        ),
    ).sortby("time")


def id_val_dict_from_table(table: pa.Table, *attr_fields: str, **renamed_attr_fields: str):
    idval_da = xr.DataArray(
        _from_chunked_array(table["val"]),
        dims="time",
        coords={
            "time": ("time", _time_from_table(table)),
            "id": ("time", _from_chunked_array(table["id"])),
            **{k: ("time", _from_chunked_array(table[k])) for k in attr_fields},
            **{k: ("time", _from_chunked_array(table[t_k])) for k, t_k in renamed_attr_fields.items()},
        },
    )
    idval_da_dict = {
        k: _unique_squeezed_attr_or_drop(da.drop_vars("id").sortby("time").rename(k))
        for k, da in idval_da.groupby("id")
    }
    return idval_da_dict


def id_val_ds_from_table(table: pa.Table):
    return (
        xr.DataArray(
            _from_chunked_array(table["val"]),
            dims="stacked",
            coords={
                "time": ("stacked", _time_from_table(table)),
                "id": ("stacked", _from_chunked_array(table["id"])),
            },
        )
        .set_index(stacked=["id", "time"])
        .to_unstacked_dataset("id")
        .sortby("time")
    )


def patient_monitor_da_dict_from_table(table: pa.Table):
    return id_val_dict_from_table(table)


def vent_ds_from_table(table: pa.Table):
    ds = id_val_ds_from_table(table)
    ds = ds.rename_vars(
        {"AFlo": "flow", "APre": "pressure", "Vol": "volume", "PHASE": "device_phase", "CO2m": "co2"}
    ).drop_vars("CO2p")
    ds["co2"] = ds["co2"].assign_attrs(units="mmHg")
    return ds


def vent_n_ds_from_table(table: pa.Table):
    return id_val_ds_from_table(table)


# FIXME: sort by `full_seq_num`, it's not modulo
def nk_dict_to_ecg_da(nk_dict: dict[str, xr.DataArray]):
    # filter out the non ecg data from nk_dict
    nk_ecg_dict = {k: v for k, v in nk_dict.items() if "ECG" in k}
    # rename each ecg(per lead) data like 'MDC_ECG_ELEC_POTL_II' -> 'II' / 'MDC_ECG_ELEC_POTL_III' -> 'III'
    nk_ecg_das = [v.rename(k.removeprefix("MDC_ECG_ELEC_POTL_")) for k, v in nk_ecg_dict.items()]
    # merge each ecg(per lead) into a new array with `lead` as a new coordinate
    nk_ecg_da = xr.merge(nk_ecg_das).to_array("lead")
    # name the data to ecg
    return nk_ecg_da.rename("ecg")


NK_RENAMES = {"SpO2": "Pleth", "ART": "Arterial", "RIMP": "RespImp"}


def nk_dict_to_ds(nk_dict: dict[str, xr.DataArray]):
    nk_ds = xr.merge([v.rename(k.removeprefix("MDC_")) for k, v in nk_dict.items() if "ECG" not in k])
    nk_ds = nk_ds.rename_vars({k: r for k, r in NK_RENAMES.items() if k in nk_ds})
    for k in nk_ds:
        if k == "RespImp":
            nk_ds[k].attrs.update({"long_name": "Respiratory Impendence"})
        elif k == "SIQ1":
            nk_ds[k].attrs.update({"long_name": "SpO2 quality"})
    return nk_ds
