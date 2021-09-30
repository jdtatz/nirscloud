import os
from pathlib import PurePath
import requests
import numpy as np
import gssapi
from requests_gssapi import HTTPSPNEGOAuth
from hdfs import Client
from requests import Session
from io import BytesIO
import pandas as pd
import xarray as xr
from .nirscloud_mongo import Meta, FastrakMeta, NIRSMeta


KAFKA_TOPICS_FT = 'fastrak2_s'
KAFKA_TOPICS_N = 'metaox_nirs_rs'
KAFKA_TOPICS_FP = 'finapres_waveform2_s' #XXX use finapres_waveform_s if you can't find any record

HDFS_NAMESERVICE = 'BabyNIRSHDFS'
HDFS_PREFIX = PurePath("/nirscloud/dedup")
HDFS_PREFIX_FT = HDFS_PREFIX / KAFKA_TOPICS_FT
HDFS_PREFIX_N = HDFS_PREFIX / KAFKA_TOPICS_N
HDFS_PREFIX_FP = HDFS_PREFIX / KAFKA_TOPICS_FP

HDFS_MASTERS = {'name': 'ceph2', 'host': 'ceph2.babynirs.org'}, {'name': 'babynirs2', 'host': 'babynirs2.babynirs.org'}, {'name': 'hdfs1', 'host': 'hdfs1.babynirs.org'}
HDFS_HTTPS_PORT = 9870
HDFS_HTTP_PORT = 9871

SPARK_KERBEROS_KEYTAB = "/home/jovyan/.spark/spark.keytab"


# TODO Verify webhdfs client impl
def create_webhfs_client(spark_kerberos_principal=None, *, proxies={}, headers={}) -> Client:
    if spark_kerberos_principal is None:
        babynirs_username = os.environ["JUPYTERHUB_USER"]
        spark_kerberos_principal = f'{babynirs_username}@BABYNIRS.ORG'
    url = ";".join(f"https://{m['host']}:{HDFS_HTTPS_PORT}" for m in HDFS_MASTERS)
    # name = gssapi.Name(spark_kerberos_principal, gssapi.NameType.user)
    # creds = gssapi.Credentials(name=name, usage="initiate")
    name = gssapi.Name(spark_kerberos_principal, gssapi.NameType.kerberos_principal)
    creds = gssapi.Credentials.acquire(name, store={"keytab": SPARK_KERBEROS_KEYTAB}).creds
    session = Session()
    session.auth = HTTPSPNEGOAuth(creds=creds)
    session.proxies.update(proxies)
    session.headers.update(headers)
    return Client(url, session=session)


def read_data_from_meta(client: Client, meta: Meta, hdfs_prefix: PurePath):
    def read_data(path: PurePath):
        with client.read(path) as reader:
            return pd.read_parquet(BytesIO(reader.read()))

    full_path = hdfs_prefix / meta.hdfs
    if client.status(full_path)["type"] == "FILE":
        return read_data(full_path)
    dfs = (read_data(PurePath(path) / file) for path, _, files in client.walk(full_path) for file in files if file.endswith(".parquet"))
    df = pd.concat(dfs)
    df.reset_index(inplace=True, drop=True)
    return df


def _to_datetime_scalar(v, unit='D'):
    if isinstance(v, np.generic):
        return v.astype(f"datetime64[{unit}]")
    else:
        return np.datetime64(v, unit)


def fastrak_ds_from_raw_df(df: pd.DataFrame, meta: FastrakMeta, position_is_in_inches: bool = True):
    from scipy.spatial.transform import Rotation

    orientation = Rotation.from_euler('ZYX', df[['a', 'e', 'r']].to_numpy(), degrees=True).as_quat()[..., [3, 0, 1, 2]]
    return xr.Dataset(
        {
            "idx": ("time", df["idx"]),
            # nirscloud metadata doesn't include units, but fastrak is probably giving inches
            "position": (("time", "cartesian_axes"), (df[["x", "y", "z"]] * 2.54) if position_is_in_inches else df[["x", "y", "z"]], dict(units="cm")),
            "orientation": (("time", "quaternion_axes"), orientation),
            # "orientation": ("time", quaternion.as_quat_array(orientation)),
        },
        coords={
            "time": ("time", df["_nano_ts"].astype("datetime64[ns]")),
            "cartesian_axes": ("cartesian_axes", ["x", "y", "z"]),
            "quaternion_axes": ("quaternion_axes", ["w", "x", "y", "z"]),
            "study": meta.study,
            "subject": meta.subject,
            "session": meta.session,
            "measurement": meta.measurement,
            "date": _to_datetime_scalar(meta.date),
            "note_id": meta.note,
            "meta_id": meta.meta,
        },
    )


def _fix_nested_array(array: np.ndarray):
    if array.dtype == np.object_:
        return np.stack([_fix_nested_array(a) for a in array])
    else:
        return array


def nirs_ds_from_raw_df(df: pd.DataFrame, meta: NIRSMeta):
    start = _to_datetime_scalar(df["_nano_ts"].min(), "ns")
    end = _to_datetime_scalar(df["_nano_ts"].max(), "ns")
    meta_dur = pd.to_timedelta(np.round(np.float64(meta.duration)), 's').to_numpy() if meta.duration is not None else None
    dt_dur = np.round((end - start) / np.timedelta64(1, 's')).astype("timedelta64[s]")
    return xr.Dataset(
        {
            "ac": (("time", "wavelength", "rho"), _fix_nested_array(df["ac"])),
            "phase": (("time", "wavelength", "rho"), _fix_nested_array(df["phase"]), dict(units="deg")),
            "dc": (("time", "wavelength", "rho"), _fix_nested_array(df["dc"])),
            "dark": (("time", "rho"), _fix_nested_array(df["dark"])),
            "aux": (("time", "rho"), _fix_nested_array(df["aux"])),
        },
        coords={
            "time": ("time", df["_offset_nano_ts"].astype("timedelta64[ns]")),
            "nirs_start_time": start,
            "nirs_end_time": end,
            "duration": meta_dur if meta_dur is not None and not np.isnat(meta_dur) else dt_dur,
            "wavelength": ("wavelength", meta.nirs_wavelengths, dict(units="nm")),
            "rho": ("rho", meta.nirs_distances, dict(units="cm")),
            "study": meta.study,
            "subject": meta.subject,
            "session": meta.session,
            "measurement": meta.measurement,
            "date": _to_datetime_scalar(meta.date),
            "note_id": meta.note,
            "meta_id": meta.meta,
        },
    )


def read_fastrak_ds_from_meta(client: Client, meta: FastrakMeta, position_is_in_inches: bool = True):
    return fastrak_ds_from_raw_df(read_data_from_meta(client, meta, HDFS_PREFIX_FT), meta, position_is_in_inches=position_is_in_inches)


def read_nirs_ds_from_meta(client: Client, meta: NIRSMeta):
    return nirs_ds_from_raw_df(read_data_from_meta(client, meta, HDFS_PREFIX_N), meta)
