from pathlib import PosixPath
from typing import Optional
import warnings

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


@warnings.deprecated("Use `nirscloud_raw.read_nirsraw` instead")
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


def add_meta_coords(ds: xr.Dataset, meta: Meta, *, nirs_det_dim: str = "rho", metaox_to_rel_time: bool = True):
    loc, trial = _maybe_rsplit_once(meta.measurement, "_")
    trial = None if trial is None else int(trial)
    session = int(meta.session.removeprefix("S")) if meta.session else None

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
        if "time" in ds.coords and np.issubdtype(ds.coords["time"].dtype, np.datetime64):
            start, end = ds["time"][0].item(), ds["time"][-1].item()
            if meta.nirs_start is not None:
                if start > meta.nirs_start:
                    start = meta.nirs_start
                elif start < meta.nirs_start:
                    warnings.warn(f"meta.nirs_start {meta.nirs_start} is later than the first data timestamp {start}")
            if meta.nirs_end is not None:
                if end < meta.nirs_end:
                    end = meta.nirs_end
                elif end > meta.nirs_end:
                    warnings.warn(f"meta.nirs_end {meta.nirs_end} is earlier than the last data timestamp {end}")
            dt_dur = np.round((end - start) / np.timedelta64(1, "s")).astype("timedelta64[s]")
            coords.update(
                nirs_start_time=start,
                nirs_end_time=end,
                duration=meta.duration if meta.duration is not None else dt_dur,
            )
            if metaox_to_rel_time:
                ds["time"] = ds["time"] - ds["time"][0]
        else:
            if meta.duration is not None:
                coords["duration"] = meta.duration
            if meta.nirs_start is not None:
                coords["nirs_start_time"] = meta.nirs_start
            if meta.nirs_end is not None:
                coords["nirs_end_time"] = meta.nirs_end

            if "nirs_start_time" in ds.attrs:
                start = ds.attrs["nirs_start_time"]
                if meta.nirs_start is not None and meta.nirs_start > start:
                    warnings.warn(f"meta.nirs_start {meta.nirs_start} is later than the first data timestamp {start}")
                    coords["nirs_start_time"] = start
                elif meta.nirs_start is None:
                    coords["nirs_start_time"] = start

        coords["rho"] = nirs_det_dim, np.array(meta.nirs_distances), {"units": "cm"}
        if meta.nirs_hz is not None:
            coords["frequency"] = (), np.array(meta.nirs_hz), {"units": "Hz"}
        if meta.nirs_wavelengths is not None:
            coords["wavelength"] = "wavelength", np.array(meta.nirs_wavelengths), {"units": "nm"}
        if meta.gains is not None:
            coords["gain"] = nirs_det_dim, np.array(meta.gains)
        ds["phase"].attrs["units"] = "radian" if meta.is_radian else "deg"
    elif isinstance(meta, DCSMeta):
        if "time" in ds.coords and np.issubdtype(ds.coords["time"].dtype, np.datetime64):
            start, end = ds["time"][0], ds["time"][-1]
            if meta.dcs_start is not None:
                if start > meta.dcs_start:
                    start = meta.dcs_start
                elif start < meta.dcs_start:
                    warnings.warn(f"meta.dcs_start {meta.dcs_start} is later than the fist data timestamp {start}")
            if meta.dcs_end is not None:
                if end < meta.dcs_end:
                    end = meta.dcs_end
                elif end > meta.dcs_end:
                    warnings.warn(f"meta.dcs_end {meta.dcs_end} is earlier than the last data timestamp {end}")
            dt_dur = np.round((end - start) / np.timedelta64(1, "s")).astype("timedelta64[s]")
            coords.update(
                dcs_start_time=start,
                dcs_end_time=end,
                duration=meta.duration if meta.duration is not None else dt_dur,
            )
            if metaox_to_rel_time:
                ds["time"] = ds["time"] - ds["time"][0]
        else:
            if meta.duration is not None:
                coords["duration"] = meta.duration
            if meta.dcs_start is not None:
                coords["dcs_start_time"] = meta.dcs_start
            if meta.dcs_end is not None:
                coords["dcs_end_time"] = meta.dcs_end

            if "dcs_start_time" in ds.attrs:
                start = ds.attrs["dcs_start_time"]
                if meta.dcs_start is not None and meta.dcs_start > start:
                    warnings.warn(f"meta.dcs_start {meta.dcs_start} is later than the first data timestamp {start}")
                    coords["dcs_start_time"] = start
                elif meta.dcs_start is None:
                    coords["dcs_start_time"] = start

        coords["rho"] = "channel", np.array(meta.dcs_distances), {"units": "cm"}
        if meta.dcs_hz is not None:
            coords["frequency"] = (), np.array(meta.dcs_hz), {"units": "Hz"}
        if meta.dcs_wavelength is not None:
            coords["wavelength"] = (), np.array(meta.dcs_wavelength), {"units": "nm"}
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


def _offset_time_from_table(table: pa.Table):
    if "_offset_nano_ts" in table.column_names and "_nano_ts" in table.column_names:
        time = _from_chunked_array(table["_offset_nano_ts"]).astype("timedelta64[ns]")
        ## TODO: loading the entire array and checking it is overkill
        ts = _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]")
        start_ts_vals = np.unique_values(ts - time)
        if start_ts_vals.size > 1:
            ## NOTE: this should never happen
            raise ValueError("Inconsistent timestamp vs timedelta")
        (start_ts,) = start_ts_vals
        return time, start_ts
    elif "_offset_nano_ts" in table.column_names:
        return _from_chunked_array(table["_offset_nano_ts"]).astype("timedelta64[ns]"), None
    elif "_nano_ts" in table.column_names:
        return _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]"), None
    ## TODO: find if "_offset_milli_ts" is used anywhere
    elif "_offset_milli_ts" in table.column_names and "_milli_ts" in table.column_names:
        time = _from_chunked_array(table["_offset_milli_ts"]).astype("timedelta64[ms]")
        ## TODO: loading the entire array and checking it is overkill
        ts = _from_chunked_array(table["_milli_ts"]).astype("datetime64[ms]")
        start_ts_vals = np.unique_values(ts - time)
        if start_ts_vals.size > 1:
            ## NOTE: this should never happen
            raise ValueError("Incomsistent timestamp vs timedelta")
        (start_ts,) = start_ts_vals
        return time, start_ts
    elif "_offset_milli_ts" in table.column_names:
        return _from_chunked_array(table["_offset_milli_ts"]).astype("timedelta64[ms]"), None
    elif "_milli_ts" in table.column_names:
        return _from_chunked_array(table["_milli_ts"]).astype("datetime64[ms]"), None
    else:
        raise KeyError(
            f"No known time column, `_offset_nano_ts`, `_nano_ts` or `_milli_ts`, found in the table. [{table.column_names}]"
        )


def nirs_ds_from_table(table: pa.Table, *, nirs_det_dim: str = "rho"):
    time, start = _offset_time_from_table(table)
    ds = (
        xr.Dataset(
            data_vars={
                "ac": (("time", "wavelength", nirs_det_dim), _from_chunked_array(table["ac"])),
                "phase": (("time", "wavelength", nirs_det_dim), _from_chunked_array(table["phase"])),
                "dc": (("time", "wavelength", nirs_det_dim), _from_chunked_array(table["dc"])),
                "dark": (("time", nirs_det_dim), _from_chunked_array(table["dark"])),
                "aux": (("time", nirs_det_dim), _from_chunked_array(table["aux"])),
            },
            coords={"time": ("time", time)},
        )
        .transpose()
        .sortby("time")
    )
    if start is not None:
        ds.attrs["nirs_start_time"] = start
    return ds


def dcs_ds_from_table(table: pa.Table, *, flipped_banks: Optional[bool] = None):
    tau = _from_chunked_array(table["t"])
    # assert np.unique(tau, axis=0).shape[0] == 1
    time, start = _offset_time_from_table(table)
    ds = (
        xr.Dataset(
            data_vars={
                "counts": (("time", "channel"), _from_chunked_array(table["CPS"]), {"units": "Hz"}),
                "value": (("time", "tau", "channel"), _from_chunked_array(table["t_val"])),
            },
            coords={
                "time": ("time", time),
                "tau": ("tau", tau[0], {"units": "s"}),
            },
        )
        .transpose()
        .sortby("time")
    )
    if start is not None:
        ds.attrs["dcs_start_time"] = start
    if flipped_banks is not None:
        if flipped_banks:
            assert ds.sizes["channel"] == 8
            ds = ds.roll({"channel": 4})
        ## NOTE: the channel coordinate is only added if it's known if the banks were flipped or not
        # ds = ds.assign_coords(channel=(["channel"], np.arange(1, 1 + ds.sizes["channel"])))
        ds = ds.assign_coords(
            xr.Coordinates.from_xindex(xr.indexes.PandasIndex(pd.RangeIndex(1, 1 + ds.sizes["channel"]), "channel"))
        )
    return ds


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
        data_vars={
            "position": (("cartesian_axes", "stacked"), position),
            "orientation": (("quaternion_axes", "stacked"), orientation),
        },
        coords={
            "idx": ("stacked", _from_chunked_array(table["idx"])),
            "time": ("stacked", _time_from_table(table)),
            "cartesian_axes": ("cartesian_axes", list(cartesian_axes)),
            "quaternion_axes": ("quaternion_axes", list(quaternion_axes)),
        },
    )
    ds = ds.set_index(stacked=["time", "idx"]).unstack("stacked").transpose("idx", ..., "time")
    return ds.sortby("time")


def finapres_ds_from_table(table: pa.Table):
    return xr.Dataset(
        data_vars={
            "pressure": ("time", _from_chunked_array(table["pressure_mmHg"]), {"units": "mmHg"}),
            # "height": ("time", _from_chunked_array(table["height_mmHg"]), {"units": "mmHg"}),
            "plethysmograph": ("time", _from_chunked_array(table["plethysmograph"])),
        },
        coords={
            "time": ("time", _time_from_table(table)),
        },
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
