import os
import requests
import numpy as np
import gssapi
from requests_gssapi import HTTPSPNEGOAuth
from hdfs import Client
from requests import Session
from fastparquet import ParquetFile
import pandas as pd
import xarray as xr


kafka_topics_FT = 'fastrak2_s'
kafka_topics_N = 'metaox_nirs_rs'
kafka_topics_FP = 'finapres_waveform2_s' #XXX use finapres_waveform_s if you can't find any record

hdfs_nameservice = 'BabyNIRSHDFS'
hdfs_prefix = "/nirscloud/dedup"
hdfs_prefix_FT = f'{hdfs_prefix}/{kafka_topics_FT}'
hdfs_prefix_N = f'{hdfs_prefix}/{kafka_topics_N}'
hdfs_prefix_FP = f'{hdfs_prefix}/{kafka_topics_FP}'

hdfs_masters = {'name': 'ceph2', 'host': 'ceph2.babynirs.org'}, {'name': 'babynirs2', 'host': 'babynirs2.babynirs.org'}, {'name': 'hdfs1', 'host': 'hdfs1.babynirs.org'}
hdfs_https_port = 9870
hdfs_http_port = 9871

park_kerberos_keytab = "/home/jovyan/.spark/spark.keytab"


# TODO Verify webhdfs client impl
def create_webhfs_client(spark_kerberos_principal=None, *, proxies={}, headers={}):
    if spark_kerberos_principal is None:
        babynirs_username = os.environ["JUPYTERHUB_USER"]
        spark_kerberos_principal = f'{babynirs_username}@BABYNIRS.ORG'
    url = ";".join(f"https://{m['host']}:{hdfs_https_port}" for m in hdfs_masters)
    name = gssapi.Name(spark_kerberos_principal, gssapi.NameType.user)
    creds = gssapi.Credentials(name=name, usage="initiate")
    session = Session()
    session.auth = HTTPSPNEGOAuth(creds=creds)
    session.proxies.update(proxies)
    session.headers.update(headers)
    return Client(url, session=session)


def read_data_from_meta(client, meta, hdfs_prefix):
    from pathlib import Path
    full_path = Path(hdfs_prefix) / meta["hdfs_path"]
    with client.read(full_path) as reader:
        # contents = reader.read()
        df = ParquetFile(reader).to_pandas()
    return df


def _to_datetime_scalar(v, unit='D'):
    if isinstance(v, np.generic):
        return v.astype(f"datetime64[{unit}]")
    else:
        return np.datetime64(v, unit)


def read_fastrak_ds_from_meta(client, meta_dict):
    from scipy.spatial.transform import Rotation

    df = read_data_from_meta(client, meta_dict, hdfs_prefix_FT)
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

def read_nirs_ds_from_meta(client, meta_dict):
    df = read_data_from_meta(client, meta_dict, hdfs_prefix_N)
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
