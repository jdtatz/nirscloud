from pathlib import PurePosixPath

import httpx
import fsspec
import fsspec.asyn

from .webhdfs import WebHDFS
from .webhdfs_schemas import fileStatusProperties


def _standarize_status(path: PurePosixPath, status: fileStatusProperties):
    suffix = status.pop("pathSuffix", None)
    if suffix:
        path = path / suffix
    status["name"] = str(path)
    status["type"] = status["type"].lower()
    status["size"] = status.pop("length")
    return status


class SyncWebHDFS(fsspec.AbstractFileSystem):
    def __init__(self, client: WebHDFS[httpx.Client],  *args, **storage_options):
        self.client = client
        super().__init__(*args, **storage_options)


class AsyncWebHDFS(fsspec.asyn.AsyncFileSystem):
    """Read-Only"""

    def __init__(self, client: WebHDFS[httpx.AsyncClient], *args, asynchronous=False, loop=None, batch_size=None, **kwargs):
        self.client = client
        super().__init__(*args, asynchronous=asynchronous, loop=loop, batch_size=batch_size, **kwargs)

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        params = {}
        offset = 0
        if start is not None:
            params["offset"] = offset = start
        if end is not None:
            params["length"] = end - offset
        return await self.client.open(PurePosixPath(path), params)

    async def _info(self, path, **kwargs):
        info = await self.client.status(path)
        if info is None:
            raise FileNotFoundError(path)
        return _standarize_status(path, info["FileStatus"])

    async def _ls(self, path, detail=True, **kwargs):
        ls = await self.client.list(path)
        statuses = ls["FileStatuses"]["FileStatus"]
        statuses = map(_standarize_status, statuses)
        if detail:
            return sorted(statuses, key=lambda s: s["name"])
        else:
            return sorted(s["name"] for s in statuses)
