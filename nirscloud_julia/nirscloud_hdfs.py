import asyncio
import os
from pathlib import PurePosixPath
from typing import Optional

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pds

from .nirscloud_mongo import Meta
from .webhdfs import WebHDFS, async_walk, sync_walk

KAFKA_TOPICS_FT = "fastrak2_s"
KAFKA_TOPICS_FT_CM = "fastrak_cm_s"
KAFKA_TOPICS_N = "metaox_nirs_rs"
KAFKA_TOPICS_D = "metaox_dcs_s"
KAFKA_TOPICS_FP = "finapres_waveform_s"
KAFKA_TOPICS_FP_2 = "finapres_waveform2_s"
KAFKA_TOPICS_FP_3 = "finapres_waveform_su"
KAFKA_TOPICS_PM = "ixtrend_waves5_s"
KAFKA_TOPICS_PM_N = "ixtrend_numerics2_s"

FASTRAK_KAFKA_TOPICS = KAFKA_TOPICS_FT_CM, KAFKA_TOPICS_FT
NIRS_KAFKA_TOPICS = (KAFKA_TOPICS_N,)
DCS_KAFKA_TOPICS = (KAFKA_TOPICS_D,)
FINAPRES_KAFKA_TOPICS = KAFKA_TOPICS_FP_3, KAFKA_TOPICS_FP_2, KAFKA_TOPICS_FP
PM_KAFKA_TOPICS = "ixtrend_waves5_s", "ixtrend_waves4_s", "ixtrend_waves3_s", "ixtrend_waves2_s", "ixtrend_waves"
PM_N_KAFKA_TOPICS = "ixtrend_numerics2_s", "ixtrend_numerics_s", "ixtrend_numerics"

HDFS_NAMESERVICE = "BabyNIRSHDFS"
HDFS_PREFIX_DEDUP = PurePosixPath("/nirscloud/dedup")
HDFS_PREFIX_AGG_HR = PurePosixPath("/nirscloud/agg_by_hr")
HDFS_PREFIX_AGG_DAY = PurePosixPath("/nirscloud/agg_by_day")
HDFS_PREFIX_AGG = PurePosixPath("/nirscloud/agg")
HDFS_PREFIX_AGG_HR = PurePosixPath("/nirscloud/agg_by_hr")
HDFS_PREFIX_AGG_DAY = PurePosixPath("/nirscloud/agg_by_day")
HDFS_PREFIX_KAFKA_TOPICS = PurePosixPath("/kafka/topics")
HDFS_PREFIX = HDFS_PREFIX_AGG

HDFS_PREFIX_FT = HDFS_PREFIX_AGG / KAFKA_TOPICS_FT
HDFS_PREFIX_FT_CM = HDFS_PREFIX_AGG / KAFKA_TOPICS_FT_CM
HDFS_PREFIX_N = HDFS_PREFIX_DEDUP / KAFKA_TOPICS_N
HDFS_PREFIX_D = HDFS_PREFIX_DEDUP / KAFKA_TOPICS_D
HDFS_PREFIX_FP = HDFS_PREFIX_DEDUP / KAFKA_TOPICS_FP
HDFS_PREFIX_FP_2 = HDFS_PREFIX_AGG / KAFKA_TOPICS_FP_2
HDFS_PREFIX_FP_3 = HDFS_PREFIX_AGG / KAFKA_TOPICS_FP_3
HDFS_PREFIX_PM = HDFS_PREFIX_AGG / KAFKA_TOPICS_PM
HDFS_PREFIX_PM_N = HDFS_PREFIX_AGG / KAFKA_TOPICS_PM_N

HDFS_FASTRAK_PREFIXES = HDFS_PREFIX_FT_CM, HDFS_PREFIX_FT, HDFS_PREFIX_DEDUP / "fastrak_s"
HDFS_NIRS_PREFIXES = (HDFS_PREFIX_N,)
HDFS_DCS_PREFIXES = (HDFS_PREFIX_D,)
HDFS_FINAPRES_PREFIXES = HDFS_PREFIX_FP_3, HDFS_PREFIX_FP_2, HDFS_PREFIX_FP
HDFS_PM_PREFIXES = (HDFS_PREFIX_PM, *(HDFS_PREFIX_DEDUP / t for t in PM_KAFKA_TOPICS))
HDFS_PM_N_PREFIXES = (HDFS_PREFIX_PM_N, *(HDFS_PREFIX_DEDUP / t for t in PM_N_KAFKA_TOPICS))


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
SPARK_KERBEROS_PRINCIPAL: Optional[str]
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


def nirscloud_webhdfs_auth(
    kerberos_principal=SPARK_KERBEROS_PRINCIPAL,
    credential_store=dict(keytab=SPARK_KERBEROS_KEYTAB),
) -> httpx.Auth:
    import gssapi
    from httpx_gssapi import HTTPSPNEGOAuth

    name = gssapi.Name(kerberos_principal, gssapi.NameType.kerberos_principal)
    creds = gssapi.Credentials.acquire(name, store=credential_store).creds
    return HTTPSPNEGOAuth(creds=creds)


def create_webhdfs_client(
    kerberos_principal=SPARK_KERBEROS_PRINCIPAL,
    credential_store=dict(keytab=SPARK_KERBEROS_KEYTAB),
    hdfs_masters=HDFS_MASTERS,
    *,
    proxies=None,
    headers={},
    limits=httpx.Limits(),
) -> WebHDFS[httpx.Client]:
    session = httpx.Client(http2=True, limits=limits, proxies=proxies)
    session.auth = nirscloud_webhdfs_auth(kerberos_principal, credential_store)
    session.headers.update(headers)
    return WebHDFS(session, *hdfs_masters, port=HDFS_HTTPS_PORT)


def create_webhfs_client(
    kerberos_principal=SPARK_KERBEROS_PRINCIPAL,
    credential_store=dict(keytab=SPARK_KERBEROS_KEYTAB),
    hdfs_masters=HDFS_MASTERS,
    *,
    proxies={},
    headers={},
) -> WebHDFS[httpx.Client]:
    from warnings import warn

    warn(
        "Call to deprecated function `create_webhfs_client`, use the correctly spelled `create_webhdfs_client`.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return create_webhdfs_client(kerberos_principal, credential_store, hdfs_masters, headers=headers)


def create_async_webhdfs_client(
    kerberos_principal=SPARK_KERBEROS_PRINCIPAL,
    credential_store=dict(keytab=SPARK_KERBEROS_KEYTAB),
    hdfs_masters=HDFS_MASTERS,
    *,
    proxies=None,
    headers={},
    limits=httpx.Limits(),
) -> WebHDFS[httpx.AsyncClient]:
    session = httpx.AsyncClient(http2=True, limits=limits, proxies=proxies)
    session.auth = nirscloud_webhdfs_auth(kerberos_principal, credential_store)
    # session.proxies.update(proxies)
    session.headers.update(headers)
    return WebHDFS(session, *hdfs_masters, port=HDFS_HTTPS_PORT)


_parquet_format = pds.ParquetFileFormat()


def _sync_read_fragment(client: WebHDFS[httpx.Client], path: PurePosixPath) -> pds.FileFragment:
    return _parquet_format.make_fragment(pa.BufferReader(client.open(path)))


async def _async_read_fragment(client: WebHDFS[httpx.AsyncClient], path: PurePosixPath) -> pds.FileFragment:
    return _parquet_format.make_fragment(pa.BufferReader(await client.open(path)))


def sync_read_fragments(client: WebHDFS[httpx.Client], dir_path: PurePosixPath) -> pds.Dataset:
    fragments = (
        _sync_read_fragment(client, PurePosixPath(path) / file)
        for path, _, files in sync_walk(client, dir_path)
        for file in files
        if file.endswith(".parquet")
    )
    return pds.FileSystemDataset(fragments, schema=fragments[0].physical_schema, format=_parquet_format)


async def async_read_fragments(client: WebHDFS[httpx.AsyncClient], dir_path: PurePosixPath) -> pds.Dataset:
    tree = await async_walk(client, dir_path)
    fragments = await asyncio.gather(
        *(
            _async_read_fragment(client, PurePosixPath(path) / file)
            for path, _, files in tree
            for file in files
            if file.endswith(".parquet")
        )
    )
    return pds.FileSystemDataset(fragments, schema=fragments[0].physical_schema, format=_parquet_format)


def read_table_from_parts(client: WebHDFS[httpx.Client], dir_path: PurePosixPath) -> pa.Table:
    status = client.status(dir_path)
    if status is None:
        raise FileNotFoundError
    if status["FileStatus"]["type"] == "FILE":
        return _sync_read_fragment(client, dir_path).to_table()
    return sync_read_fragments(client, dir_path).to_table()


def read_table_from_meta(client: WebHDFS[httpx.Client], meta: Meta, *hdfs_prefix_options: PurePosixPath) -> pa.Table:
    for hdfs_prefix in hdfs_prefix_options:
        status = client.status(hdfs_prefix / meta.hdfs)
        if status is not None:
            break
    else:
        raise FileNotFoundError
    full_path = hdfs_prefix / meta.hdfs
    if status["FileStatus"]["type"] == "FILE":
        return _sync_read_fragment(client, full_path).to_table()
    return sync_read_fragments(client, full_path).to_table()


async def async_read_table_from_parts(client: WebHDFS[httpx.AsyncClient], dir_path: PurePosixPath) -> pa.Table:
    status = await client.status(dir_path)
    if status is None:
        raise FileNotFoundError
    if status["FileStatus"]["type"] == "FILE":
        return (await _async_read_fragment(client, dir_path)).to_table()
    return (await async_read_fragments(client, dir_path)).to_table()


async def async_read_table_from_meta(
    client: WebHDFS[httpx.AsyncClient], meta: Meta, *hdfs_prefix_options: PurePosixPath
) -> pa.Table:
    for hdfs_prefix in hdfs_prefix_options:
        status = await client.status(hdfs_prefix / meta.hdfs)
        if status is not None:
            break
    else:
        raise FileNotFoundError
    full_path = hdfs_prefix / meta.hdfs
    if status["FileStatus"]["type"] == "FILE":
        return (await _async_read_fragment(client, full_path)).to_table()
    return (await async_read_fragments(client, full_path)).to_table()
