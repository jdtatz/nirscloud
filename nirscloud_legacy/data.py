import warnings
from collections.abc import Sequence
from functools import partial
from typing import Literal, Optional

import numpy as np
import pyarrow as pa
import xarray as xr
from metaox_parser import fix_flipped_banks
from scipy.spatial.transform import Rotation

from .mongo import (
    DCSMeta,
    FastrakMeta,
    FinapresMeta,
    Meta,
    NIRSMeta,
    PatientMonitorMeta,
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


def add_meta_coords(ds: xr.Dataset, meta: Meta, *, metaox_to_rel_time: bool = True):
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

        coords["rho"] = "detector", np.array(meta.nirs_distances), {"units": "cm"}
        if meta.nirs_hz is not None:
            coords["frequency"] = (), np.array(meta.nirs_hz), {"units": "Hz"}
        if meta.nirs_wavelengths is not None:
            coords["wavelength"] = "wavelength", np.array(meta.nirs_wavelengths), {"units": "nm"}
        if meta.gains is not None:
            coords["gain"] = "detector", np.array(meta.gains)
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
        # TODO: timezone fixup
        pass
    elif isinstance(meta, PatientMonitorMeta):
        # TODO
        pass
    else:
        # raise TypeError
        pass
    return ds.assign_coords(coords)


def _pa_scalar_shape(v: pa.Scalar):
    if isinstance(v, Sequence):
        return (len(v), *_pa_scalar_shape(v[0]))
    else:
        return ()


def _from_chunked_array(carray: pa.ChunkedArray) -> np.ndarray:
    # TODO: remove this workaround once `pa.StringArray.to_numpy()` no longer yields an object array
    if pa.types.is_string(carray.type):
        return np.array(carray.to_pylist())
    if pa.types.is_nested(carray.type):
        shape = (len(carray), *_pa_scalar_shape(carray[0]))
        return carray.combine_chunks().flatten(recursive=True).to_numpy().reshape(shape)
    else:
        return carray.to_numpy()


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


def _start_from_ts_td(ts, dt, *, prefer_rela: bool = True):
    start_ts_vals = np.unique_values(ts - dt)
    if start_ts_vals.size > 1:
        ## NOTE: this should never happen
        raise ValueError("Inconsistent timestamp vs timedelta")
    (start_ts,) = start_ts_vals
    return dt if prefer_rela else ts, start_ts


def _offset_time_from_table(table: pa.Table, *, prefer_rela: bool = True):
    if "_offset_nano_ts" in table.column_names and "_nano_ts" in table.column_names:
        ## TODO: loading both arrays in their entirety for checking is overkill
        return _start_from_ts_td(
            _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]"),
            _from_chunked_array(table["_offset_nano_ts"]).astype("timedelta64[ns]"),
            prefer_rela=prefer_rela,
        )
    elif "_offset_nano_ts" in table.column_names:
        return _from_chunked_array(table["_offset_nano_ts"]).astype("timedelta64[ns]"), None
    elif "_nano_ts" in table.column_names:
        return _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]"), None
    ## TODO: find if "_offset_milli_ts" is used anywhere
    elif "_offset_milli_ts" in table.column_names and "_milli_ts" in table.column_names:
        return _start_from_ts_td(
            _from_chunked_array(table["_milli_ts"]).astype("datetime64[ms]"),
            _from_chunked_array(table["_offset_milli_ts"]).astype("timedelta64[ms]"),
            prefer_rela=prefer_rela,
        )
    elif "_offset_milli_ts" in table.column_names:
        return _from_chunked_array(table["_offset_milli_ts"]).astype("timedelta64[ms]"), None
    elif "_milli_ts" in table.column_names:
        return _from_chunked_array(table["_milli_ts"]).astype("datetime64[ms]"), None
    else:
        raise KeyError(
            f"No known time column, `_offset_nano_ts`, `_nano_ts` or `_milli_ts`, found in the table. [{table.column_names}]"
        )


def nirs_ds_from_table(table: pa.Table):
    time, start = _offset_time_from_table(table)
    ds = (
        xr.Dataset(
            data_vars={
                "ac": (("time", "wavelength", "detector"), _from_chunked_array(table["ac"])),
                "phase": (("time", "wavelength", "detector"), _from_chunked_array(table["phase"])),
                "dc": (("time", "wavelength", "detector"), _from_chunked_array(table["dc"])),
                "dark": (("time", "detector"), _from_chunked_array(table["dark"])),
                "aux": (("time", "detector"), _from_chunked_array(table["aux"])),
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
    return fix_flipped_banks(ds, flipped_banks=flipped_banks)


def fastrak_raw_stacked_ds_from_table(table: pa.Table, *, prefer_timedelta: bool = False):
    # idx=0 is always the pen, idx=1 is always the nirs sensor, and if idx=2 exists then it's the head sensor (refrence point for dual-quat transformation)

    # Fasktrak may be at 60HZ, but our data has large gaps, so a modifed timedelta RangeIndex isn't applicable
    time, start = _offset_time_from_table(table, prefer_rela=prefer_timedelta)
    ds = xr.Dataset(
        data_vars={c: (("stacked",), _from_chunked_array(table[c])) for c in ("x", "y", "z", "a", "e", "r")},
        coords={
            "idx": ("stacked", _from_chunked_array(table["idx"])),
            "list_id": ("stacked", _from_chunked_array(table["list_id"])),
            "time": ("stacked", time),
        },
    )
    if start is not None:
        ds.attrs["fastrak_start_time"] = start
    return ds


def _stack_dataset_vars(ds: xr.Dataset, *dvars: str, dim: str, axis: Literal[0, -1] = 0):
    if dim in ds.dims:
        raise ValueError(f"{dim!r} is not a new dim")
    if axis not in (0, -1):
        raise NotImplementedError
    if len(dvars) == 0:
        raise ValueError("no variables specified")

    ds = ds[list(dvars)]
    if axis == 0:
        stacked = ds.to_dataarray(dim)
        # FIXME: `xr.Dataset.to_dataarray` only uses the Datasets attrs instead of the variables attrs
        stacked = stacked.drop_attrs().assign_attrs(**ds[dvars[0]].attrs)
    else:
        stacked = ds.to_stacked_array("variable", ds.dims, dim)
        # FIXME: `xr.DataArray.unstack` modifies the array's data even if the multi-index has only a single level
        # stacked = stacked.unstack("variable")
        stacked = stacked.reset_index("variable").set_xindex(dim).swap_dims({"variable": dim})
    assert tuple(stacked.coords[dim].values) == dvars
    return stacked


def _euler_to_quat(seq: str, angles, *, degrees: bool = False, canonical: bool = False, scalar_first: bool = False):

    return Rotation.from_euler(seq, angles, degrees=degrees).as_quat(canonical=canonical, scalar_first=scalar_first)


def convert_raw_fastrak_ds(
    raw_ds: xr.Dataset, *, canonical: bool = False, scalar_first: bool = False, transpose: bool = False
):
    position = _stack_dataset_vars(raw_ds, "x", "y", "z", dim="cartesian_axes", axis=0 if transpose else -1)
    orientation = xr.apply_ufunc(
        partial(_euler_to_quat, "ZYX", degrees=True, canonical=canonical, scalar_first=scalar_first),
        _stack_dataset_vars(raw_ds, "a", "e", "r", dim="euler_axes", axis=-1),
        input_core_dims=(("euler_axes",),),
        output_core_dims=(("quaternion_axes",),),
        dask="allowed",
        keep_attrs=False,
    ).assign_coords(
        quaternion_axes=(("quaternion_axes",), ["w", "x", "y", "z"] if scalar_first else ["x", "y", "z", "w"])
    )
    if transpose:
        orientation = orientation.transpose("quaternion_axes", ...)
    return xr.Dataset({"position": position, "orientation": orientation})


def fastrak_stacked_ds_from_table(table: pa.Table, *, prefer_timedelta: bool = False, scalar_first: bool = False):
    # idx=0 is always the pen, idx=1 is always the nirs sensor, and if idx=2 exists then it's the head sensor (refrence point for dual-quat transformation)
    cartesian_axes = "x", "y", "z"
    euler_axes = "a", "e", "r"
    quaternion_axes = ("w", "x", "y", "z") if scalar_first else ("x", "y", "z", "w")

    # Fasktrak may be at 60HZ, but our data has large gaps, so a modifed timedelta RangeIndex isn't applicable
    time, start = _offset_time_from_table(table, prefer_rela=prefer_timedelta)
    position = np.stack([_from_chunked_array(table[c]) for c in cartesian_axes], axis=0)
    angles = np.stack([_from_chunked_array(table[c]) for c in euler_axes], axis=1)
    orientation = _euler_to_quat("ZYX", angles, degrees=True, scalar_first=scalar_first).T
    ds = xr.Dataset(
        data_vars={
            "position": (("cartesian_axes", "stacked"), position),
            "orientation": (("quaternion_axes", "stacked"), orientation),
        },
        coords={
            "idx": ("stacked", _from_chunked_array(table["idx"])),
            "list_id": ("stacked", _from_chunked_array(table["list_id"])),
            "time": ("stacked", time),
            "cartesian_axes": ("cartesian_axes", list(cartesian_axes)),
            "quaternion_axes": ("quaternion_axes", list(quaternion_axes)),
        },
    )
    if start is not None:
        ds.attrs["fastrak_start_time"] = start
    return ds


def unstack_fastrak_stacked_ds(
    stacked_ds: xr.Dataset,
    *,
    keep: Literal["first", "last"] = "first",
    join: Literal["outer", "inner", "exact"] = "outer",
):
    n0 = stacked_ds.sizes["stacked"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Try using swap_dims instead.*")
        deduped_grps = [
            ds.drop_vars("idx")
            .assign_coords(idx=idx)
            .sortby(["time", "list_id"])
            # FIXME: this will warn and suggest swap_dims, but that's broken for this use-case https://github.com/pydata/xarray/issues/8646
            .rename({"stacked": "time"})
            .set_xindex("time")
            .drop_duplicates("time", keep=keep)
            .drop_vars("list_id")
            for idx, ds in stacked_ds.groupby("idx")
        ]
    n2 = sum(d.sizes["time"] for d in deduped_grps)
    ds = xr.concat(deduped_grps, "idx", coords=[], join=join, compat="identical", combine_attrs="identical")
    if n0 > n2:
        warnings.warn(f"{n0 - n2} duplicate time points were dropped", stacklevel=2)
    # TODO: is a seconding sorting required?
    return ds.sortby("time")


def fastrak_ds_from_table(
    table: pa.Table,
    *,
    scalar_first: bool = True,
    keep: Literal["first", "last"] = "first",
    join: Literal["outer", "inner", "exact"] = "outer",
):
    stacked_ds = fastrak_stacked_ds_from_table(table, scalar_first=scalar_first)
    return unstack_fastrak_stacked_ds(stacked_ds, keep=keep, join=join)


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
