import asyncio
from dataclasses import dataclass, field
from functools import partialmethod, singledispatch
from inspect import isawaitable
from itertools import chain
from pathlib import PurePosixPath
from types import MethodType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from httpx import (
    URL,
    AsyncClient,
    Client,
    HTTPStatusError,
    NetworkError,
    Request,
    RequestError,
    Response,
    TimeoutException,
)
from httpx._client import BaseClient
from typing_extensions import (
    Awaitable,
    Concatenate,
    Literal,
    LiteralString,
    NotRequired,
    ParamSpec,
    TypedDict,
    Unpack,
    overload,
)

from .webhdfs_schemas import FileStatus, FileStatuses, RemoteException

# https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/WebHDFS.html#JSON_Schemas


T = TypeVar("T")
R = TypeVar("R", covariant=True)

# C = TypeVar("C", bound=BaseClient)
# C = TypeVar("C", bound=Union[Client, AsyncClient])
C = TypeVar("C", Client, AsyncClient)


class BaseHDFSRemoteException(Exception):
    message: str
    """Exception message"""
    java_class_name: Optional[str]
    """Java class name of the exception"""

    def __init__(self, remote_error: RemoteException):
        self.message = remote_error["RemoteException"]["message"]
        self.java_class_name = remote_error["RemoteException"].get("javaClassName")
        super().__init__(self.message)


class HDFSRemoteException(BaseHDFSRemoteException):
    exception: str
    """Name of the exception"""

    def __init__(self, remote_error: RemoteException):
        self.exception = remote_error["RemoteException"]["exception"]
        super().__init__(remote_error)


class HDFSRemoteRetriableException(BaseHDFSRemoteException):
    pass


class HDFSRemoteStandbyException(BaseHDFSRemoteException):
    pass


def _from_remote_exception(remote_error: RemoteException):
    exception = remote_error["RemoteException"]["exception"]
    message = remote_error["RemoteException"]["message"]
    if exception in ("IllegalArgumentException", "UnsupportedOperationException"):
        return ValueError(message)
    elif exception in ("SecurityException", "AccessControlException", "IOException"):
        return PermissionError(message)
    elif exception == "RetriableException":
        return HDFSRemoteRetriableException(remote_error)
    elif exception == "StandbyException":
        return HDFSRemoteStandbyException(remote_error)
    else:
        return HDFSRemoteException(remote_error)


async def _async_map(f: Callable[[T], R], v: Awaitable[T]) -> R:
    return f(await v)


def _handle_response(response: Response) -> Optional[Response]:
    if response.is_success:
        return response
    elif response.status_code == 404:
        return None
    elif response.headers.get("content-type") == "application/json":
        remote_error: RemoteException = response.json()
        raise _from_remote_exception(remote_error)
    else:
        response.raise_for_status()
        raise AssertionError("Expected code to be unreachable")


class SendKwargs(TypedDict):
    stream: NotRequired[bool]
    # auth: NotRequired[Any]
    follow_redirects: NotRequired[bool]


@singledispatch
def _failover_send(
    client: BaseClient, request: Request, *failover_requests: Request, **send_kwargs: Unpack[SendKwargs]
):
    raise NotImplementedError


@_failover_send.register
def _sync_failover_send(
    client: Client, request: Request, *failover_requests: Request, **send_kwargs: Unpack[SendKwargs]
) -> Optional[Response]:
    try:
        return _handle_response(client.send(request, **send_kwargs))
    except (TimeoutException, NetworkError, HDFSRemoteRetriableException, HDFSRemoteStandbyException):
        if not failover_requests:
            raise
    return _sync_failover_send(client, *failover_requests, **send_kwargs)


@_failover_send.register
async def _async_failover_send(
    client: AsyncClient, request: Request, *failover_requests: Request, **send_kwargs: Unpack[SendKwargs]
) -> Optional[Response]:
    try:
        return _handle_response(await client.send(request, **send_kwargs))
    except (TimeoutException, NetworkError, HDFSRemoteRetriableException, HDFSRemoteStandbyException):
        if not failover_requests:
            raise
    return await _async_failover_send(client, *failover_requests, **send_kwargs)


@overload
def _failover_send_and_map(
    client: Client,
    handle: Callable[[Optional[Response]], R],
    request: Request,
    *failover_requests: Request,
    **send_kwargs: Unpack[SendKwargs],
) -> R:
    ...


@overload
def _failover_send_and_map(
    client: AsyncClient,
    handle: Callable[[Optional[Response]], R],
    request: Request,
    *failover_requests: Request,
    **send_kwargs: Unpack[SendKwargs],
) -> Awaitable[R]:
    ...


def _failover_send_and_map(
    client: C,
    handle: Callable[[Optional[Response]], R],
    request: Request,
    *failover_requests: Request,
    **send_kwargs: Unpack[SendKwargs],
) -> Union[R, Awaitable[R]]:
    r = _failover_send(client, request, *failover_requests, **send_kwargs)
    if isawaitable(r):
        return _async_map(handle, r)
    else:
        return handle(r)


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


BuildRequestParams = ParamSpec("BuildRequestParams")
# # bind only the kwargs to `BuildRequestParams`
# _helper: Callable[BuildRequestParams, Request] = partial(BaseClient().build_request, "GET", "127.0.0.1")


class BuildRequestKwargs(TypedDict):
    content: NotRequired[Union[str, bytes]]
    json: NotRequired[Any]
    timeout: NotRequired[float]


@dataclass
class WebHDFSOperation(Generic[R]):
    handle: Callable[[Optional[Response]], R]
    method: Literal["GET", "PUT", "POST", "DELETE"]
    operation: LiteralString
    params: Dict[str, Any] = field(default_factory=dict)
    # FIXME: mypy doesn't understand that a `TypedDict` subclass can be used as a type constructer
    # send_kwargs: SendKwargs = field(default_factory=SendKwargs)
    send_kwargs: SendKwargs = field(default_factory=lambda: SendKwargs())

    @overload
    def __call__(
        self,
        wrapped: "WebHDFS[Client]",
        path: PurePosixPath,
        params: Optional[Dict[str, Any]] = None,
        **build_request_kwargs: Unpack[BuildRequestKwargs],
    ) -> R:
        ...

    @overload
    def __call__(
        self,
        wrapped: "WebHDFS[AsyncClient]",
        path: PurePosixPath,
        params: Optional[Dict[str, Any]] = None,
        **build_request_kwargs: Unpack[BuildRequestKwargs],
    ) -> Awaitable[R]:
        ...

    def __call__(
        self,
        wrapped: "WebHDFS[C]",
        path: PurePosixPath,
        params: Optional[Dict[str, Any]] = None,
        **build_request_kwargs: Unpack[BuildRequestKwargs],
    ):
        requests = wrapped._build_requests(
            self.method, self.operation, path, {**self.params, **(params or {})}, build_request_kwargs
        )
        return _failover_send_and_map(wrapped.client, self.handle, *requests, **self.send_kwargs)

    # FIXME: can't add kwargs typing to `Callable`
    @overload
    def __get__(
        self, instance: "WebHDFS[Client]", owner=None
    ) -> Callable[Concatenate[PurePosixPath, BuildRequestParams], R]:
        ...

    @overload
    def __get__(
        self, instance: "WebHDFS[AsyncClient]", owner=None
    ) -> Callable[Concatenate[PurePosixPath, BuildRequestParams], Awaitable[R]]:
        ...

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return MethodType(self, instance)


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

    def _build_requests(
        self,
        method: Literal["GET", "PUT", "POST", "DELETE"],
        operation: LiteralString,
        path: PurePosixPath,
        params: Dict[str, Any],
        build_request_kwargs: BuildRequestKwargs,
    ) -> List[Request]:
        if path.is_absolute():
            path = path.relative_to(path.root)
        requests = [
            self.client.build_request(
                method, base_url.join(str(path)), params={"op": operation, **params}, **build_request_kwargs
            )
            for base_url in self._base_urls
        ]
        return requests

    # # TODO: Use once `partial`/`partialmethod` typing is fixed, see https://github.com/python/mypy/issues/1484
    # def _operation(
    #     self,
    #     handle: Callable[[Optional[Response]], R],
    #     method: Literal["GET", "PUT", "POST", "DELETE"],
    #     operation: LiteralString,
    #     path: PurePosixPath,
    #     params: Optional[Dict[str, Any]] = None,
    #     send_kwargs: Optional[SendKwargs] = None,
    #     **build_request_kwargs: Unpack[BuildRequestKwargs],
    # ): # -> Union[R, Awaitable[R]]:
    #     requests = self._build_requests(method, operation, path, params or {}, **build_request_kwargs)
    #     return _failover_send_and_map(self.client, handle, *requests, **(send_kwargs or {}))
    # open = partialmethod(_operation, _handle_open_response, "GET", "OPEN", send_kwargs=dict(follow_redirects=True))
    # status = partialmethod(_operation, _handle_file_status_response, "GET", "GETFILESTATUS")
    # list = partialmethod(_operation, cast(Callable[[Optional[Response]], FileStatuses], _handle_json_response), "GET", "LISTSTATUS")

    open = WebHDFSOperation(_handle_open_response, "GET", "OPEN", send_kwargs=dict(follow_redirects=True))
    status = WebHDFSOperation(_handle_file_status_response, "GET", "GETFILESTATUS")
    list = WebHDFSOperation(
        cast(Callable[[Optional[Response]], FileStatuses], _handle_json_response), "GET", "LISTSTATUS"
    )


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
