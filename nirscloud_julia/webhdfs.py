import asyncio
from itertools import chain
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

from httpx import (
    URL,
    AsyncClient,
    Client,
    NetworkError,
    Request,
    Response,
    TimeoutException,
)
from typing_extensions import Awaitable, Literal, LiteralString, overload

from .webhdfs_schemas import FileStatus, FileStatuses, RemoteException

# https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/WebHDFS.html#JSON_Schemas


T = TypeVar("T")
R = TypeVar("R")

# C = TypeVar("C", bound=BaseClient)
# C = TypeVar("C", bound=Union[Client, AsyncClient])
C = TypeVar("C", Client, AsyncClient)


class HDFSRemoteException(Exception):
    exception: str
    """Name of the exception"""
    message: str
    """Exception message"""
    javaClassName: Optional[str]
    """Java class name of the exception"""

    def __init__(self, remote_error: RemoteException):
        self.message = remote_error["RemoteException"]["message"]
        self.exception = remote_error["RemoteException"]["exception"]
        self.javaClassName = remote_error["RemoteException"].get("javaClassName")
        super().__init__(self.message)


async def _async_map(f: Callable[[T], R], v: Awaitable[T]) -> R:
    return f(await v)


def _handle_response(response: Response) -> Optional[Response]:
    if response.is_success:
        return response
    elif response.status_code == 404:
        return None
    elif response.headers.get("content-type") == "application/json":
        remote_error: RemoteException = response.json()
        raise HDFSRemoteException(remote_error)
    else:
        response.raise_for_status()
        assert False


def _sync_failover_send(
    client: Client, request: Request, *failover_requests: Request, **send_kwargs
) -> Optional[Response]:
    try:
        return _handle_response(client.send(request, **send_kwargs))
    except (TimeoutException, NetworkError):
        if not failover_requests:
            raise
    except HDFSRemoteException as e:
        if e.exception not in ("RetriableException", "StandbyException") or not failover_requests:
            raise
    return _sync_failover_send(client, *failover_requests, **send_kwargs)


async def _async_failover_send(
    client: AsyncClient, request: Request, *failover_requests: Request, **send_kwargs
) -> Optional[Response]:
    try:
        return _handle_response(await client.send(request, **send_kwargs))
    except (TimeoutException, NetworkError):
        if not failover_requests:
            raise
    except HDFSRemoteException as e:
        if e.exception not in ("RetriableException", "StandbyException") or not failover_requests:
            raise
    return await _async_failover_send(client, *failover_requests, **send_kwargs)


@overload
def _failover_send_and_map(
    client: Client,
    handle: Callable[[Optional[Response]], R],
    request: Request,
    *failover_requests: Request,
    **send_kwargs,
) -> R:
    ...


@overload
def _failover_send_and_map(
    client: AsyncClient,
    handle: Callable[[Optional[Response]], R],
    request: Request,
    *failover_requests: Request,
    **send_kwargs,
) -> Awaitable[R]:
    ...


def _failover_send_and_map(
    client: C, handle: Callable[[Optional[Response]], R], request: Request, *failover_requests: Request, **send_kwargs
) -> Union[R, Awaitable[R]]:
    if isinstance(client, AsyncClient):
        return _async_map(handle, _async_failover_send(client, request, *failover_requests, **send_kwargs))
    else:
        return handle(_sync_failover_send(client, request, *failover_requests, **send_kwargs))


def _handle_open_response(response: Optional[Response]) -> bytes:
    if response is None:
        raise FileNotFoundError
    return response.content


def _handle_json_response(response: Optional[Response]):
    if response is None:
        raise FileNotFoundError
    return response.json()


def _handle_file_status_response(response: Optional[Response]) -> Optional[FileStatus]:
    return None if response is None else response.json()


class WebHDFS(Generic[C]):
    client: C

    def __init__(
        self,
        client: C,
        *hosts: str,
        port: Optional[int] = None,
        transport: str = "https",
        endpoint: str = "webhdfs",
        api_version: str = "v1",
    ):
        self.client = client
        assert len(hosts) > 0
        self._base_urls = [URL(f"{transport}://{host}/{endpoint}/{api_version}/") for host in hosts]
        if port is not None:
            self._base_urls = [url.copy_with(port=port) for url in self._base_urls]

    # FIXME remove overloads in WebHDFS when type infrence for `_failover_send_and_map` finally works
    @overload
    def _operation(
        self: "WebHDFS[Client]",
        handle: Callable[[Optional[Response]], R],
        method: Literal["GET", "PUT", "POST", "DELETE"],
        operation: LiteralString,
        path: PurePosixPath,
        params: Optional[Dict[str, Any]] = None,
        send_kwargs: Optional[Dict[str, Any]] = None,
        **extras,
    ) -> R:
        ...

    @overload
    def _operation(
        self: "WebHDFS[AsyncClient]",
        handle: Callable[[Optional[Response]], R],
        method: Literal["GET", "PUT", "POST", "DELETE"],
        operation: LiteralString,
        path: PurePosixPath,
        params: Optional[Dict[str, Any]] = None,
        send_kwargs: Optional[Dict[str, Any]] = None,
        **extras,
    ) -> Awaitable[R]:
        ...

    def _operation(
        self,
        handle: Callable[[Optional[Response]], R],
        method: Literal["GET", "PUT", "POST", "DELETE"],
        operation: LiteralString,
        path: PurePosixPath,
        params: Optional[Dict[str, Any]] = None,
        send_kwargs: Optional[Dict[str, Any]] = None,
        **extras,
    ) -> Union[R, Awaitable[R]]:
        if path.is_absolute():
            path = path.relative_to(path.root)
        requests = [
            self.client.build_request(
                method, base_url.join(str(path)), params={"op": operation, **(params or {})}, **extras
            )
            for base_url in self._base_urls
        ]
        return _failover_send_and_map(self.client, handle, *requests, **(send_kwargs or {}))

    @overload
    def open(self: "WebHDFS[Client]", path: PurePosixPath) -> bytes:
        ...

    @overload
    def open(self: "WebHDFS[AsyncClient]", path: PurePosixPath) -> Awaitable[bytes]:
        ...

    def open(self, path: PurePosixPath):
        return self._operation(_handle_open_response, "GET", "OPEN", path, send_kwargs=dict(follow_redirects=True))

    @overload
    def status(self: "WebHDFS[Client]", path: PurePosixPath) -> Optional[FileStatus]:
        ...

    @overload
    def status(self: "WebHDFS[AsyncClient]", path: PurePosixPath) -> Awaitable[Optional[FileStatus]]:
        ...

    def status(self, path: PurePosixPath):
        return self._operation(_handle_file_status_response, "GET", "GETFILESTATUS", path)

    @overload
    def list(self: "WebHDFS[Client]", path: PurePosixPath) -> FileStatuses:
        ...

    @overload
    def list(self: "WebHDFS[AsyncClient]", path: PurePosixPath) -> Awaitable[FileStatuses]:
        ...

    def list(self, path: PurePosixPath):
        handle: Callable[[Optional[Response]], FileStatuses] = _handle_json_response
        return self._operation(handle, "GET", "LISTSTATUS", path)


def sync_walk(client: WebHDFS[Client], dirpath: PurePosixPath) -> List[Tuple[PurePosixPath, List[str], List[str]]]:
    children = client.list(dirpath)["FileStatuses"]["FileStatus"]
    # called walk on a file
    if len(children) == 1 and children[0]["pathSuffix"] == "":
        return []
    dirnames = [c["pathSuffix"] for c in children if c.get("type") == "DIRECTORY"]
    filenames = [c["pathSuffix"] for c in children if c.get("type") == "FILE"]
    l0 = dirpath, dirnames, filenames
    L = [sync_walk(client, dirpath / name) for name in dirnames]
    return [l0, *chain.from_iterable(L)]


async def async_walk(
    client: WebHDFS[AsyncClient], dirpath: PurePosixPath
) -> List[Tuple[PurePosixPath, List[str], List[str]]]:
    children = (await client.list(dirpath))["FileStatuses"]["FileStatus"]
    # called walk on a file
    if len(children) == 1 and children[0]["pathSuffix"] == "":
        return []
    dirnames = [c["pathSuffix"] for c in children if c.get("type") == "DIRECTORY"]
    filenames = [c["pathSuffix"] for c in children if c.get("type") == "FILE"]
    l0 = dirpath, dirnames, filenames
    L = await asyncio.gather(*(async_walk(client, dirpath / name) for name in dirnames))
    return [l0, *chain.from_iterable(L)]
