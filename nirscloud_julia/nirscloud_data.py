from pathlib import PosixPath

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from .nirscloud_mongo import (
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


def add_meta_coords(ds: xr.Dataset, meta: Meta):
    coords = {
        "study": meta.study,
        "subject": meta.subject,
        "session": meta.session,
        "device": meta.device,
        "measurement": meta.measurement,
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
        )
        if meta.nirs_wavelengths is not None:
            coords["wavelength"] = "wavelength", np.array(meta.nirs_wavelengths), dict(units="nm")
        if meta.gains is not None:
            coords["gain"] = "rho", np.array(meta.gains)
        ds["time"] = ds["time"] - ds["time"][0]
        ds = ds.assign(phase=ds["phase"].assign_attrs(units="radian" if meta.is_radian else "deg"))
    elif isinstance(meta, FastrakMeta):
        position = ds["position"].assign_attrs(units="cm")
        if not meta.is_cm:
            position = position * 2.54
        ds = ds.assign(position=position)
    elif isinstance(meta, FinapresMeta):
        pass
    elif isinstance(meta, PatientMonitorMeta):
        pass
    else:
        # raise TypeError
        pass
    return ds.assign_coords(coords)


def _scalar_dtype(pa_type: pa.DataType):
    if hasattr(pa_type, "value_type"):
        return _scalar_dtype(pa_type.value_type)
    return pa_type.to_pandas_dtype()


def _from_chunked_array(carray: pa.ChunkedArray) -> np.ndarray:
    return (
        np.array(carray.to_pylist(), dtype=_scalar_dtype(carray.type))
        if pa.types.is_nested(carray.type)
        else carray.to_numpy()
    )


def nirs_ds_from_table(table: pa.Table):
    return xr.Dataset(
        data_vars=dict(
            ac=(("time", "wavelength", "rho"), _from_chunked_array(table["ac"])),
            phase=(("time", "wavelength", "rho"), _from_chunked_array(table["phase"])),
            dc=(("time", "wavelength", "rho"), _from_chunked_array(table["dc"])),
            dark=(("time", "rho"), _from_chunked_array(table["dark"])),
            aux=(("time", "rho"), _from_chunked_array(table["aux"])),
        ),
        coords=dict(time=("time", _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]"))),
    ).transpose(..., "time")


def fastrak_ds_from_table(table: pa.Table):
    from scipy.spatial.transform import Rotation

    cartesian_axes = "x", "y", "z"
    euler_axes = "a", "e", "r"
    quaternion_axes = "w", "x", "y", "z"

    position = np.stack([_from_chunked_array(table[c]) for c in cartesian_axes], axis=0)
    angles = np.stack([_from_chunked_array(table[c]) for c in euler_axes], axis=1)
    orientation = np.roll(Rotation.from_euler("ZYX", angles, degrees=True).as_quat().transpose(), 1, axis=0)
    stacked = pd.MultiIndex.from_arrays(
        [_from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]"), _from_chunked_array(table["idx"])],
        names=["time", "idx"],
    )
    ds = xr.Dataset(
        data_vars=dict(
            # idx=("time", _from_chunked_array(table["idx"])),
            position=(("cartesian_axes", "stacked"), position),
            orientation=(("quaternion_axes", "stacked"), orientation),
        ),
        coords=dict(
            stacked=stacked,
            # time=("time", _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]")),
            cartesian_axes=("cartesian_axes", list(cartesian_axes)),
            quaternion_axes=("quaternion_axes", list(quaternion_axes)),
        ),
    )
    # ks, vs = zip(*ds.groupby("idx"))
    # ds = xr.concat([v.drop_vars("idx") for v in vs], xr.DataArray(list(ks), dims="idx"))
    ds = ds.unstack("stacked").transpose("idx", ..., "time")
    return ds.sortby("time")


def finapres_ds_from_table(table: pa.Table):
    return xr.Dataset(
        data_vars=dict(
            pressure=("time", _from_chunked_array(table["pressure_mmHg"]), dict(units="mmHg")),
            # height=("time", _from_chunked_array(table["height_mmHg"]), dict(units="mmHg")),
            plethysmograph=("time", _from_chunked_array(table["plethysmograph"])),
        ),
        coords=dict(
            time=("time", _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]")),
        ),
    )


def patient_monitor_da_dict_from_table(table: pa.Table):
    return {
        pm_id: ds.drop_vars("id").sortby("time").assign_coords({"id": pm_id})
        for pm_id, ds in xr.DataArray(
            _from_chunked_array(table["val"]),
            dims="time",
            coords=dict(
                time=("time", _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]")),
                id=("time", _from_chunked_array(table["id"])),
            ),
        ).groupby("id")
    }
