import os
import time
import logging
from pathlib import Path
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import quaternion

import nirscloud_util_meta
import nirscloud_util_hdfs

from .dual_quaternion import DualQuaternion

nirscloud_meta_logger = logging.getLogger('nirscloud_util_meta')
nirscloud_hdfs_logger = logging.getLogger('nirscloud_util_hdfs')
hdfs_logger = logging.getLogger('hdfs')


def set_logging_level(level):
    nirscloud_meta_logger.setLevel(level)
    nirscloud_hdfs_logger.setLevel(level)
    hdfs_logger.setLevel(level)

set_logging_level(logging.WARNING)

hostname = 'mongos.mongo.svc.cluster.local:27017'
pemkeyfile = '/etc/mongo/jhub-keypem.pem'
sslca = '/etc/mongo/root-ca.pem'

kafka_topics_FT = 'fastrak2_s'
kafka_topics_N = 'metaox_nirs_rs'
kafka_topics_FP = 'finapres_waveform2_s' #XXX use finapres_waveform_s if you can't find any record

hdfs_nameservice = 'BabyNIRSHDFS'
hdfs_prefix = "/nirscloud/dedup"
hdfs_prefix_FT = f'{hdfs_prefix}/{kafka_topics_FT}'
hdfs_prefix_N = f'{hdfs_prefix}/{kafka_topics_N}'
hdfs_prefix_FP = f'{hdfs_prefix}/{kafka_topics_FP}'

babynirs_username = os.environ["JUPYTERHUB_USER"]
spark_kerberos_principal = f'{babynirs_username}@BABYNIRS.ORG'


def init_meta():
    return nirscloud_util_meta.init(nirscloud_meta_logger, 'meta', hostname=hostname, ssl=True, cert=pemkeyfile, ca=sslca)


def init_hdfs():
    params = {
        'spark_kerberos_principal': spark_kerberos_principal,
    }
    return nirscloud_util_hdfs.init('/etc/jhub/conf/production.ini', params=params)


def init():
    init_meta()
    init_hdfs()



core_schema_fields = "_id", "meta_id"
commmon_schema_fields = "group_id", "measurement_id", "note_id", "postfix_id", "session_id", "study_id", "subject_id", "the_date", "hdfs_path", "file_prefix"
fastrak_schema_fields = (*commmon_schema_fields,)
# *StartNanoTS and *EndNanoTS are not always there and aren't needed
nirs_schema_fields = (*commmon_schema_fields, "nirsDistances", "nirsWavelengths", "nirs_hz", "duration",) #"nirsStartNanoTS", "nirsEndNanoTS",)
dcs_schema_fields  = (*commmon_schema_fields, "dcsDistances", "dcsWavelength", "gains", "dcs_hz",) #"dcsStartNanoTS", "dcsEndNanoTS")


class NIRSCloudError(Exception):
    pass


class NIRSCloudMongoError(NIRSCloudError):
    pass


def query_meta(query, fields=None):
    err, db_results = nirscloud_util_meta.query_meta(query, fields)
    if err:
        if type(err) is Exception:
            raise NIRSCloudMongoError(*err.args)
        else:
            raise err
    return db_results


def query_fastrak_meta(query={}):
    return query_meta({"n_fastrak_dedup.val": {"$exists": True}, **query}, fields=["meta_id", *fastrak_schema_fields])


def query_nirs_meta(query={}):
    return query_meta({"n_nirs_dedup.val":    {"$exists": True}, **query}, fields=["meta_id", *nirs_schema_fields])



class NIRSCloudHDFSError(NIRSCloudError):
    pass


def read_data_from_meta(meta, hdfs_prefix):
    full_path = Path(hdfs_prefix) / meta["hdfs_path"]
    err, df = nirscloud_util_hdfs.from_hdfs_path(str(full_path), False)
    if err:
        if type(err) is Exception:
            raise NIRSCloudHDFSError(*err.args)
        else:
            raise err
    return df

def read_data_from_meta_list(meta_list, hdfs_prefix):
    return pd.concat([read_data_from_meta(meta, hdfs_prefix) for meta in meta_list])

def convert_nano_ts_to_index_inplace(df: pd.DataFrame, timezone: Optional[str] = None):
    df.set_index("_nano_ts", inplace=True)
    time_type = f"datetime64[ns, {timezone}]" if timezone is not None else "datetime64[ns]"
    df.index = df.index.astype(time_type, copy=False)
    df.index.rename("time", inplace=True)


def _to_datetime_scalar(v, unit='D'):
    if isinstance(v, np.generic):
        return v.astype(f"datetime64[{unit}]")
    else:
        return np.datetime64(v, unit)


def read_fastrak_ds_from_meta(meta_dict):
    from scipy.spatial.transform import Rotation

    df = read_data_from_meta(meta_dict, hdfs_prefix_FT)
    orientation = Rotation.from_euler('ZYX', df[['a', 'e', 'r']].to_numpy(), degrees=True).as_quat()[..., [3, 0, 1, 2]]
    return xr.Dataset(
        {
            "idx": ("time", df["idx"]),
            # nirscloud metadata doesn't include units, but fastrak is probably giving inches
            "position": (("time", "cartesian_axes"), df[["x", "y", "z"]] * 2.54, dict(units="cm")),
            "orientation": (("time", "quaternion_axes"), orientation),
            # "orientation": ("time", quaternion.as_quat_array(orientation)),
        },
        coords={
            "time": ("time", df["_nano_ts"].astype("datetime64[ns]")),
            "cartesian_axes": ("cartesian_axes", ["x", "y", "z"]),
            "quaternion_axes": ("quaternion_axes", ["w", "x", "y", "z"]),
            "study": meta_dict["study_id"],
            "subject": meta_dict["subject_id"],
            "session": meta_dict["session_id"],
            "measurement": meta_dict["measurement_id"],
            "date": _to_datetime_scalar(meta_dict["the_date"]),
            "note_id": meta_dict["note_id"],
            "meta_id": meta_dict["meta_id"],
        },
    )

def read_nirs_ds_from_meta(meta_dict):
    df = read_data_from_meta(meta_dict, hdfs_prefix_N)
    fix2d = lambda v: np.stack(v)
    fix3d = lambda v: np.stack([np.stack(arr) for arr in v])
    if "nirsStartNanoTS" in meta_dict and "nirsEndNanoTS" in meta_dict:
        start = _to_datetime_scalar(meta_dict["nirsStartNanoTS"], "ns")
        end = _to_datetime_scalar(meta_dict["nirsEndNanoTS"], "ns")
        assert df["_nano_ts"].min() == meta_dict["nirsStartNanoTS"]
        assert df["_nano_ts"].max() == meta_dict["nirsEndNanoTS"]
    else:
        start = _to_datetime_scalar(df["_nano_ts"].min(), "ns")
        end = _to_datetime_scalar(df["_nano_ts"].max(), "ns")
    meta_dur = pd.to_timedelta(np.round(np.float64(meta_dict["duration"])), 's').to_numpy() if "duration" in meta_dict else None
    dt_dur = np.round((end - start) / np.timedelta64(1, 's')).astype("timedelta64[s]")
    return xr.Dataset(
        {
            "ac": (("time", "wavelength", "rho"), fix3d(df["ac"].values)),
            "phase": (("time", "wavelength", "rho"), fix3d(df["phase"].values), dict(units="deg")),
            "dc": (("time", "wavelength", "rho"), fix3d(df["dc"].values)),
            "dark": (("time", "rho"), fix2d(df["dark"].values)),
            "aux": (("time", "rho"), fix2d(df["aux"].values)),
        },
        coords={
            "time": ("time", df["_offset_nano_ts"].astype("timedelta64[ns]")),
            "nirs_start_time": start,
            "nirs_end_time": end,
            "duration": meta_dur if meta_dur is not None and not np.isnat(meta_dur) else dt_dur,
            "wavelength": ("wavelength", meta_dict["nirsWavelengths"], dict(units="nm")),
            "rho": ("rho", meta_dict["nirsDistances"], dict(units="cm")),
            "study": meta_dict["study_id"],
            "subject": meta_dict["subject_id"],
            "session": meta_dict["session_id"],
            "measurement": meta_dict["measurement_id"],
            "date": _to_datetime_scalar(meta_dict["the_date"]),
            "note_id": meta_dict["note_id"],
            "meta_id": meta_dict["meta_id"],
        },
    )

def load_fastrak_fiducial_ds(note_id: str) -> xr.Dataset:
    fiducials = []
    measurement_id_prefixs = "ftmidnose", "ftleftear", "ftrightear"
    fiducial_locs = "nose", "left_ear", "right_ear"
    for measurement_id_prefix, loc in zip(measurement_id_prefixs, fiducial_locs):
        *_, fastrak_meta = query_fastrak_meta({'note_id.val': note_id, 'measurement_id.val': {"$regex": f"{measurement_id_prefix}_\\d+"},})
        fastrak_ds = read_fastrak_ds_from_meta(fastrak_meta)
        fiducial_ds = xr.concat(dict(fastrak_ds.groupby("idx")).values(), dim="fastrak_idx").squeeze("time").set_coords("idx").rename(time="fiducial_time", idx="fastrak_idx")
        fiducial_ds.coords["fiducial"] = (), loc
        fiducials.append(fiducial_ds.drop([k for k in fiducial_ds.coords.keys() if k.endswith("_id")]))
    return xr.concat(fiducials, dim="fiducial")


def load_fastrak_ds(subject_id: str, the_date: str) -> xr.Dataset:
    pencil_idx = 0
    head_idx=1
    nirs_idx=2
    measurement_id_prefix = "ftmsmt"
    *_, fastrak_meta = query_fastrak_meta({'subject_id.val': subject_id, "the_date.val": the_date, 'measurement_id.val': {"$regex": f"{measurement_id_prefix}_\\d+"},})
    fastrak_ds = read_fastrak_ds_from_meta(fastrak_meta)
    fstrk_grps = dict(fastrak_ds.groupby("idx", restore_coord_dims=True))
    fastrak_ds = xr.concat([fstrk_grps[head_idx].drop_vars("idx"), fstrk_grps[nirs_idx].drop_vars("idx"),], pd.Index(["head", "nirs"], name="location"))
    fastrak_ds = fastrak_ds.sortby("time")
    head_dq = DualQuaternion.from_rigid_position(fastrak_ds.position.sel(location="head"), quaternion.as_quat_array(fastrak_ds.orientation.sel(location="head")))
    nirs_dq = DualQuaternion.from_rigid_position(fastrak_ds.position.sel(location="nirs"), quaternion.as_quat_array(fastrak_ds.orientation.sel(location="nirs")))
    rel_dq = head_dq.inverse() * nirs_dq
    fastrak_ds = xr.concat((
        fastrak_ds,
        xr.Dataset(
            {
                "position": (("time", "cartesian_axes"), rel_dq.translational_componet()),
                "orientation": (("time", "quaternion_axes"), quaternion.as_float_array(rel_dq.real)),
            },
            coords={**fastrak_ds.coords, "location": "relative"},
        ),
    ), dim="location")
    fiducial_ds = load_fastrak_fiducial_ds(fastrak_ds.coords["note_id"].item())
    return fastrak_ds.assign_coords(fiducial_position=fiducial_ds.position, fiducial_orientation=fiducial_ds.orientation)


def load_nirs_ds(subject_id: str, the_date: str) -> xr.Dataset:
    db_results = query_nirs_meta({'subject_id.val': subject_id, 'the_date.val': the_date, 'measurement_id.val': {"$not": {"$regex": "CAL.*"}},})
    nirs_db_results = pd.DataFrame(db_results)
    nirs_dss = [read_nirs_ds_from_meta(meta).sortby("time") for _, meta in nirs_db_results.iterrows() if not meta["measurement_id"].startswith("CAL")]
    nirs_ds = xr.concat(nirs_dss, dim="measurement")
    return nirs_ds

