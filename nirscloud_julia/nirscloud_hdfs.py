import os
from pathlib import PurePath
import numpy as np
import gssapi
from requests_gssapi import HTTPSPNEGOAuth
from hdfs import Client
from requests import Session
from io import BytesIO
import pandas as pd
import xarray as xr
from .nirscloud_mongo import (
    FinapresMeta,
    Meta,
    FastrakMeta,
    NIRSMeta,
    PatientMonitorMeta,
)


KAFKA_TOPICS_FT = "fastrak2_s"
KAFKA_TOPICS_FT_CM = "fastrak_cm_s"
KAFKA_TOPICS_N = "metaox_nirs_rs"
KAFKA_TOPICS_FP = "finapres_waveform_s"
KAFKA_TOPICS_FP_2 = "finapres_waveform2_s"
KAFKA_TOPICS_FP_3 = "finapres_waveform_su"
KAFKA_TOPICS_PM = "ixtrend_waves5_s"
KAFKA_TOPICS_PM_N = "ixtrend_numerics2_s"

HDFS_NAMESERVICE = "BabyNIRSHDFS"
HDFS_PREFIX = PurePath("/nirscloud/dedup")
HDFS_PREFIX_AGG = PurePath("/nirscloud/agg")
HDFS_PREFIX_FT = HDFS_PREFIX_AGG / KAFKA_TOPICS_FT
HDFS_PREFIX_FT_CM = HDFS_PREFIX_AGG / KAFKA_TOPICS_FT_CM
HDFS_PREFIX_N = HDFS_PREFIX / KAFKA_TOPICS_N
HDFS_PREFIX_FP = HDFS_PREFIX / KAFKA_TOPICS_FP
HDFS_PREFIX_FP_2 = HDFS_PREFIX_AGG / KAFKA_TOPICS_FP_2
HDFS_PREFIX_FP_3 = HDFS_PREFIX_AGG / KAFKA_TOPICS_FP_3
HDFS_PREFIX_PM = HDFS_PREFIX_AGG / KAFKA_TOPICS_PM
HDFS_PREFIX_PM_N = HDFS_PREFIX_AGG / KAFKA_TOPICS_PM_N

PATIENT_MONITOR_COMPONENT_MAPPING = {
    0x013D: "III",
    0x0102: "II",
    0x013E: "aVR",
    0x0140: "aVF",
    0x4BB4: "Pleth",
    0x4A14: "ABP",
    0x5000: "Resp",
    0x0104: "Compound ECG-II",
    0x4182: "HR",
    0x4A05: "NBP-SYS",
    0x4A06: "NBP-DIA",
    0x4A07: "NBP-MEAN",
    0x4BB8: "SpO2",
    0x4822: "Pulse.1",
    0x500A: "RR",
    0x4BB0: "Perf-REL",
    0x480A: "Pulse",
}

HDFS_MASTERS = "hdfs1.babynirs.org", "hdfs2.babynirs.org", "hdfs4.babynirs.org"
HDFS_HTTPS_PORT = 9870
HDFS_HTTP_PORT = 9871
if "JUPYTERHUB_USER" in os.environ:
    SPARK_KERBEROS_PRINCIPAL = f"{os.environ['JUPYTERHUB_USER']}@BABYNIRS.ORG"
elif "USER" in os.environ:
    SPARK_KERBEROS_PRINCIPAL = f"{os.environ['USER']}@BABYNIRS.ORG"
elif "USERNAME" in os.environ:
    SPARK_KERBEROS_PRINCIPAL = f"{os.environ['USERNAME']}@BABYNIRS.ORG"
else:
    try:
        import getpass

        SPARK_KERBEROS_PRINCIPAL = f"{getpass.getuser()}@BABYNIRS.ORG"
    except Exception:
        # getpass.getuser doesn't specify which Exception type is raised on failure
        SPARK_KERBEROS_PRINCIPAL = None
SPARK_KERBEROS_KEYTAB = "/home/jovyan/.spark/spark.keytab"


def create_webhfs_client(
    kerberos_principal=SPARK_KERBEROS_PRINCIPAL,
    credential_store=dict(keytab=SPARK_KERBEROS_KEYTAB),
    hdfs_masters=HDFS_MASTERS,
    *,
    proxies={},
    headers={},
) -> Client:
    url = ";".join(f"https://{url}:{HDFS_HTTPS_PORT}" for url in hdfs_masters)
    name = gssapi.Name(kerberos_principal, gssapi.NameType.kerberos_principal)
    creds = gssapi.Credentials.acquire(name, store=credential_store).creds
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
    dfs = (
        read_data(PurePath(path) / file)
        for path, _, files in client.walk(full_path)
        for file in files
        if file.endswith(".parquet")
    )
    df = pd.concat(dfs)
    df.reset_index(inplace=True, drop=True)
    return df


def _to_datetime_scalar(v, unit="D"):
    if isinstance(v, np.generic):
        return v.astype(f"datetime64[{unit}]")
    else:
        return np.datetime64(v, unit)


def fastrak_ds_from_raw_df(df: pd.DataFrame, meta: FastrakMeta):
    from scipy.spatial.transform import Rotation

    position = df[["x", "y", "z"]]
    orientation = Rotation.from_euler("ZYX", df[["a", "e", "r"]].to_numpy(), degrees=True).as_quat()[..., [3, 0, 1, 2]]
    ds = xr.Dataset(
        {
            "idx": ("time", df["idx"]),
            # nirscloud metadata doesn't include units, but fastrak is probably giving inches
            "position": (
                ("time", "cartesian_axes"),
                position if meta.is_cm else (position * 2.54),
                dict(units="cm"),
            ),
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
            "device": meta.device,
            "measurement": meta.measurement,
            "date": _to_datetime_scalar(meta.date),
            "note_id": meta.note,
            "meta_id": meta.meta,
        },
    )
    ks, vs = zip(*ds.groupby("idx"))
    ds = xr.concat([v.drop_vars("idx") for v in vs], pd.Index(ks, name="idx"))
    return ds.sortby("time")


def _fix_nested_array(array: np.ndarray):
    if array.dtype == np.object_:
        return np.stack([_fix_nested_array(a) for a in array])
    else:
        return array


def nirs_ds_from_raw_df(df: pd.DataFrame, meta: NIRSMeta):
    start = _to_datetime_scalar(df["_nano_ts"].min(), "ns")
    end = _to_datetime_scalar(df["_nano_ts"].max(), "ns")
    dt_dur = np.round((end - start) / np.timedelta64(1, "s")).astype("timedelta64[s]")
    return xr.Dataset(
        {
            "ac": (("time", "wavelength", "rho"), _fix_nested_array(df["ac"])),
            "phase": (
                ("time", "wavelength", "rho"),
                _fix_nested_array(df["phase"]),
                dict(units="radian" if meta.is_radian else "deg"),
            ),
            "dc": (("time", "wavelength", "rho"), _fix_nested_array(df["dc"])),
            "dark": (("time", "rho"), _fix_nested_array(df["dark"])),
            "aux": (("time", "rho"), _fix_nested_array(df["aux"])),
        },
        coords={
            "time": ("time", df["_offset_nano_ts"].astype("timedelta64[ns]")),
            "nirs_start_time": start,
            "nirs_end_time": end,
            "duration": meta.duration if meta.duration is not None else dt_dur,
            "wavelength": (
                "wavelength",
                np.array(meta.nirs_wavelengths),
                dict(units="nm"),
            ),
            "rho": ("rho", np.array(meta.nirs_distances), dict(units="cm")),
            "gain": ("rho", np.array(meta.gains)),
            "study": meta.study,
            "subject": meta.subject,
            "session": meta.session,
            "device": meta.device,
            "measurement": meta.measurement,
            "date": _to_datetime_scalar(meta.date),
            "note_id": meta.note,
            "meta_id": meta.meta,
        },
    )


def finapres_ds_from_raw_df(df: pd.DataFrame, meta: FinapresMeta):
    return xr.Dataset(
        {
            "pressure": ("time", df["pressure_mmHg"], dict(units="mmHg")),
            # "height": ("time", df["height_mmHg"], dict(units="mmHg")),
            "plethysmograph": ("time", df["plethysmograph"]),
        },
        coords={
            "time": ("time", df["_nano_ts"].astype("datetime64[ns]")),
            "study": meta.study,
            "subject": meta.subject,
            "session": meta.session,
            "device": meta.device,
            "measurement": meta.measurement,
            "date": _to_datetime_scalar(meta.date),
            "note_id": meta.note,
            "meta_id": meta.meta,
        },
    )


def patient_monitor_ds_from_raw_df(df: pd.DataFrame, meta: PatientMonitorMeta):
    return {
        idv: xr.DataArray(
            df["val"],
            dims="time",
            coords={
                "id": idv,
                "time": ("time", df["_nano_ts"].astype("datetime64[ns]")),
                "study": meta.study,
                "subject": meta.subject,
                "session": meta.session,
                "device": meta.device,
                "measurement": meta.measurement,
                "date": _to_datetime_scalar(meta.date),
                "note_id": meta.note,
                "meta_id": meta.meta,
            },
        ).sortby("time")
        for idv, df in df.groupby("id")
    }


def read_fastrak_ds_from_meta(client: Client, meta: FastrakMeta):
    return fastrak_ds_from_raw_df(
        read_data_from_meta(
            client,
            meta,
            HDFS_PREFIX_FT_CM if meta.is_cm else HDFS_PREFIX_FT,
        ),
        meta,
    )


def read_nirs_ds_from_meta(client: Client, meta: NIRSMeta):
    return nirs_ds_from_raw_df(read_data_from_meta(client, meta, HDFS_PREFIX_N), meta)
