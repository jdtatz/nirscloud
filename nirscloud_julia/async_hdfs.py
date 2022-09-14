import asyncio
import os
from io import BytesIO
from itertools import chain
from pathlib import PurePosixPath

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from .nirscloud_mongo import Meta


class AsyncWebHDFS:
    # TODO multiple distrbuted host masters
    def __init__(
        self,
        session: httpx.AsyncClient,
        host: str,
        port: int,
        transport: str = "https",
        endpoint: str = "webhdfs",
        api_version: str = "v1",
    ):
        self._session = session
        self._base_url = f"{transport}://{host}:{port}/{endpoint}/{api_version}/"

    async def _get_op(self, path: PurePosixPath, op: str, raise_on_missing=True):
        response = await self._session.get(f"{self._base_url}{path}", params=dict(op=op))
        if response.is_success:
            return response.json()
        elif response.status_code == 404 and not raise_on_missing:
            return None
        else:
            response.raise_for_status()

    async def status(self, path: PurePosixPath):
        response_json = await self._get_op(path, "GETFILESTATUS", raise_on_missing=False)
        return response_json["FileStatus"] if response_json is not None else None

    async def list(self, dir_path: PurePosixPath):
        response_json = await self._get_op(dir_path, "LISTSTATUS")
        return response_json["FileStatuses"]["FileStatus"]

    async def open(self, fpath: PurePosixPath):
        response = await self._session.get(f"{self._base_url}{fpath}", params=dict(op="OPEN"), follow_redirects=True)
        response.raise_for_status()
        return response.content

    async def walk(self, dirpath: PurePosixPath):
        children = await self.list(dirpath)
        # called walk on a file
        if len(children) == 1 and children[0]["pathSuffix"] == "":
            return
        dirnames = [c["pathSuffix"] for c in children if c.get("type") == "DIRECTORY"]
        filenames = [c["pathSuffix"] for c in children if c.get("type") == "FILE"]
        l0 = dirpath, dirnames, filenames
        L = await asyncio.gather(*(self.walk(dirpath / name) for name in dirnames))
        return [l0, *chain.from_iterable(L)]


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


def create_async_webhdfs_client(
    kerberos_principal=SPARK_KERBEROS_PRINCIPAL,
    credential_store=dict(keytab=SPARK_KERBEROS_KEYTAB),
    hdfs_masters=HDFS_MASTERS,
    *,
    # proxies={},
    headers={},
    limits=httpx.Limits(),
) -> AsyncWebHDFS:
    import gssapi
    from httpx_gssapi import HTTPSPNEGOAuth

    name = gssapi.Name(kerberos_principal, gssapi.NameType.kerberos_principal)
    creds = gssapi.Credentials.acquire(name, store=credential_store).creds
    session = httpx.AsyncClient(http2=True, limits=limits)
    session.auth = HTTPSPNEGOAuth(creds=creds)
    # session.proxies.update(proxies)
    session.headers.update(headers)
    return AsyncWebHDFS(session=session, host=hdfs_masters[0], port=HDFS_HTTPS_PORT)


async def async_read_table_from_meta(client: AsyncWebHDFS, meta: Meta, *hdfs_prefix_options: PurePosixPath) -> pa.Table:
    async def read_data(path: PurePosixPath) -> pa.Table:
        return pq.read_table(BytesIO(await client.open(path)))

    for hdfs_prefix in hdfs_prefix_options:
        status = await client.status(hdfs_prefix / meta.hdfs)
        if status is not None:
            break
    else:
        raise FileNotFoundError
    full_path = hdfs_prefix / meta.hdfs
    if status["type"] == "FILE":
        return await read_data(full_path)
    tree = await client.walk(full_path)
    tables = await asyncio.gather(
        *(
            read_data(PurePosixPath(path) / file)
            for path, _, files in tree
            for file in files
            if file.endswith(".parquet")
        )
    )
    return pa.concat_tables(tables)
