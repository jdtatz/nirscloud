from pathlib import PosixPath

import numpy as np
import pandas as pd
import xarray as xr
from .nirscloud_hdfs import read_fastrak_ds_from_meta, read_nirs_ds_from_meta
from .nirscloud_mongo import query_fastrak_meta, query_nirs_meta
import quaternion
from .dual_quaternion import DualQuaternion

__all__ = [
    "load_fastrak_fiducial_ds",
    "load_fastrak_ds",
    "load_nirs_ds",
    "combine_measurements_ds",
]


def load_fastrak_fiducial_ds(
    mongo_client, webhdfs_client, note_id: str, position_is_in_inches: bool = True
) -> xr.Dataset:
    fiducials = []
    measurement_id_prefixs = "ftmidnose", "ftleftear", "ftrightear"
    fiducial_locs = "nose", "left_ear", "right_ear"
    for measurement_id_prefix, loc in zip(measurement_id_prefixs, fiducial_locs):
        *_, fastrak_meta = query_fastrak_meta(
            mongo_client,
            {
                "note_id.val": note_id,
                "measurement_id.val": {"$regex": f"{measurement_id_prefix}_\\d+"},
            },
        )
        fastrak_ds = read_fastrak_ds_from_meta(
            webhdfs_client, fastrak_meta, position_is_in_inches=position_is_in_inches
        )
        fiducial_ds = fastrak_ds.squeeze("time").rename(time="fiducial_time", idx="fastrak_idx")
        fiducial_ds.coords["fiducial"] = (), loc
        fiducials.append(fiducial_ds.drop([k for k in fiducial_ds.coords.keys() if k.endswith("_id")]))
    return xr.concat(fiducials, dim="fiducial")


def load_fastrak_ds(
    mongo_client,
    webhdfs_client,
    subject_id: str,
    the_date: str,
    position_is_in_inches: bool = True,
) -> xr.Dataset:
    pencil_idx = 0
    head_idx = 1
    nirs_idx = 2
    measurement_id_prefix = "ftmsmt"
    *_, fastrak_meta = query_fastrak_meta(
        mongo_client,
        {
            "subject_id.val": subject_id,
            "the_date.val": the_date,
            "measurement_id.val": {"$regex": f"{measurement_id_prefix}_\\d+"},
        },
    )
    fastrak_ds = read_fastrak_ds_from_meta(webhdfs_client, fastrak_meta, position_is_in_inches=position_is_in_inches)
    fastrak_ds = xr.concat(
        [
            fastrak_ds.sel(idx=head_idx),
            fastrak_ds.sel(idx=nirs_idx),
        ],
        pd.Index(["head", "nirs"], name="location"),
    )
    fastrak_ds = fastrak_ds.sortby("time")
    head_dq = DualQuaternion.from_rigid_position(
        fastrak_ds.position.sel(location="head"),
        quaternion.as_quat_array(fastrak_ds.orientation.sel(location="head")),
    )
    nirs_dq = DualQuaternion.from_rigid_position(
        fastrak_ds.position.sel(location="nirs"),
        quaternion.as_quat_array(fastrak_ds.orientation.sel(location="nirs")),
    )
    rel_dq = head_dq.inverse() * nirs_dq
    fastrak_ds = xr.concat(
        (
            fastrak_ds,
            xr.Dataset(
                {
                    "position": (
                        ("time", "cartesian_axes"),
                        rel_dq.translational_componet(),
                    ),
                    "orientation": (
                        ("time", "quaternion_axes"),
                        quaternion.as_float_array(rel_dq.real),
                    ),
                },
                coords={**fastrak_ds.coords, "location": "relative"},
            ),
        ),
        dim="location",
    )
    fiducial_ds = load_fastrak_fiducial_ds(
        mongo_client,
        webhdfs_client,
        fastrak_ds.coords["note_id"].item(),
        position_is_in_inches=position_is_in_inches,
    )
    return fastrak_ds.assign_coords(
        fiducial_position=fiducial_ds.position,
        fiducial_orientation=fiducial_ds.orientation,
    )


def load_nirs_ds(
    mongo_client,
    webhdfs_client,
    subject_id: str,
    the_date: str,
    with_calib: bool = True,
) -> xr.Dataset:
    query = {
        "subject_id.val": subject_id,
        "the_date.val": the_date,
    }
    if not with_calib:
        query["measurement_id.val"] = {"$not": {"$regex": "CAL.*"}}
    db_results = query_nirs_meta(mongo_client, query)
    nirs_db_results = pd.DataFrame(db_results)
    nirs_dss = [read_nirs_ds_from_meta(webhdfs_client, meta).sortby("time") for _, meta in nirs_db_results.iterrows()]
    nirs_ds = xr.concat(nirs_dss, dim="measurement")
    return nirs_ds


def combine_measurements_ds(fastrak_ds: xr.Dataset, nirs_ds: xr.Dataset) -> xr.Dataset:
    nirs_ds = nirs_ds.sel(measurement=~nirs_ds.measurement.str.startswith("CAL"))
    fastrak_measurements = []
    for dt in xr.concat((nirs_ds.nirs_start_time, nirs_ds.nirs_start_time + nirs_ds.duration), dim="t").transpose(
        "measurement", "t"
    ):
        m = str(dt.measurement.values)
        start, end = dt.values
        t_slice = slice(start, end + 1)
        fastrak_time_sliced = (
            fastrak_ds.sel(time=t_slice).drop_vars("measurement").rename_dims(time="fastrak_time")
        )  # .rename(time="fastrak_time")
        fastrak_time_sliced.coords["fastrak_time"] = fastrak_time_sliced["time"] - start
        fastrak_time_sliced.coords["measurement"] = m
        fastrak_measurements.append(fastrak_time_sliced.drop_vars("time"))
    measurements = xr.merge(
        (
            xr.concat(fastrak_measurements, dim="measurement").drop_vars(["meta_id", "session"]),
            nirs_ds.drop_vars(["meta_id", "session"]),
        )
    )
    (
        measurements.coords["measurement_location"],
        measurements.coords["measurement_trial"],
    ) = measurements.measurement.str.split("split_axis", "_", 1).transpose("split_axis", ...)
    return measurements


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
