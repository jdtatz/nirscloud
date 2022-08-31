import os
import asyncio
from itertools import chain
from pathlib import PurePath, PurePosixPath
from io import BytesIO

import httpx

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from .nirscloud_mongo import Meta, NIRSMeta, FastrakMeta, FinapresMeta, PatientMonitorMeta


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
) -> AsyncWebHDFS:
    import gssapi
    from httpx_gssapi import HTTPSPNEGOAuth

    name = gssapi.Name(kerberos_principal, gssapi.NameType.kerberos_principal)
    creds = gssapi.Credentials.acquire(name, store=credential_store).creds
    session = httpx.AsyncClient()
    session.auth = HTTPSPNEGOAuth(creds=creds)
    # session.proxies.update(proxies)
    session.headers.update(headers)
    return AsyncWebHDFS(session=session, host=hdfs_masters[0], port=HDFS_HTTPS_PORT)


async def async_read_data_from_meta(client: AsyncWebHDFS, meta: Meta, *hdfs_prefix_options: PurePath) -> pa.Table:
    async def read_data(path: PurePath):
        return pq.read_table(BytesIO(await client.open(path)))

    # Only here to silence mypy worrying about it being Unbound
    hdfs_prefix = hdfs_prefix_options[0]
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
        *(read_data(PurePath(path) / file) for path, _, files in tree for file in files if file.endswith(".parquet"))
    )
    return pa.concat_tables(tables)


def _to_datetime_scalar(v, unit="D"):
    if isinstance(v, np.generic):
        return v.astype(f"datetime64[{unit}]")
    else:
        return np.datetime64(v, unit)


def add_meta_coords(ds: xr.Dataset, meta: Meta):
    coords = {
        "study": meta.study,
        "subject": meta.subject,
        "session": meta.session,
        "device": meta.device,
        "measurement": meta.measurement,
        "date": _to_datetime_scalar(meta.date),
        "note_id": meta.note,
        "meta_id": meta.meta,
        # "group": meta.group,
    }
    if isinstance(meta, NIRSMeta):
        start, end = ds["time"][0], ds["time"][-1]
        dt_dur = np.round((end - start) / np.timedelta64(1, "s")).astype("timedelta64[s]")
        coords.update(
            nirs_start_time=start,
            nirs_end_time=end,
            duration=meta.duration if meta.duration is not None else dt_dur,
            wavelength=(
                "wavelength",
                np.array(meta.nirs_wavelengths),
                dict(units="nm"),
            ),
            rho=("rho", np.array(meta.nirs_distances), dict(units="cm")),
            gain=("rho", np.array(meta.gains)),
        )
        ds = ds.assign(phase=ds["phase"].assign_attrs(units="radian" if meta.is_radian else "deg"))
    elif isinstance(meta, FastrakMeta):
        position = ds["position"].assign_attrs(units="cm")
        if not meta.is_cm:
            position = position * 2.54
        ds = ds.assign(position=position)
    elif isinstance(meta, FinapresMeta):
        pass
    elif isinstance(meta, PatientMonitorMeta):
        pass
    else:
        # raise TypeError
        pass
    return ds.assign_coords(coords)


def _scalar_dtype(pa_type: pa.DataType):
    if hasattr(pa_type, "value_type"):
        return _scalar_dtype(pa_type.value_type)
    return pa_type.to_pandas_dtype()


def _from_chunked_array(carray: pa.ChunkedArray) -> np.ndarray:
    return (
        np.array(carray.to_pylist(), dtype=_scalar_dtype(carray.type))
        if pa.types.is_nested(carray.type)
        else carray.to_numpy()
    )


def nirs_ds_from_table(table: pa.Table):
    return xr.Dataset(
        data_vars=dict(
            ac=(("time", "wavelength", "rho"), _from_chunked_array(table["ac"])),
            phase=(("time", "wavelength", "rho"), _from_chunked_array(table["phase"])),
            dc=(("time", "wavelength", "rho"), _from_chunked_array(table["dc"])),
            dark=(("time", "rho"), _from_chunked_array(table["dark"])),
            aux=(("time", "rho"), _from_chunked_array(table["aux"])),
        ),
        coords=dict(time=("time", _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]"))),
    )


def fastrak_ds_from_table(table: pa.Table):
    from scipy.spatial.transform import Rotation

    cartesian_axes = "x", "y", "z"
    euler_axes = "a", "e", "r"
    quaternion_axes = "w", "x", "y", "z"

    position = np.stack([_from_chunked_array(table[c]) for c in cartesian_axes], axis=-1)
    angles = np.stack([_from_chunked_array(table[c]) for c in euler_axes], axis=-1)
    orientation = Rotation.from_euler("ZYX", angles, degrees=True).as_quat()[..., [3, 0, 1, 2]]
    ds = xr.Dataset(
        data_vars=dict(
            idx=("time", _from_chunked_array(table["idx"])),
            position=(("time", "cartesian_axes"), position),
            orientation=(("time", "quaternion_axes"), orientation),
        ),
        coords=dict(
            time=("time", _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]")),
            cartesian_axes=("cartesian_axes", list(cartesian_axes)),
            quaternion_axes=("quaternion_axes", list(quaternion_axes)),
        ),
    )
    ks, vs = zip(*ds.groupby("idx"))
    ds = xr.concat([v.drop_vars("idx") for v in vs], xr.DataArray(ks, dims="idx"))
    return ds.sortby("time")


def finapres_ds_from_table(table: pa.Table):
    return xr.Dataset(
        data_vars=dict(
            pressure=("time", _from_chunked_array(table["pressure_mmHg"]), dict(units="mmHg")),
            # height=("time", _from_chunked_array(table["height_mmHg"]), dict(units="mmHg")),
            plethysmograph=("time", _from_chunked_array(table["plethysmograph"])),
        ),
        coords=dict(
            time=("time", _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]")),
        ),
    )


def patient_monitor_da_dict_from_table(table: pa.Table):
    return dict(
        xr.DataArray(
            _from_chunked_array(
                table["val"],
                dims="time",
                coords=dict(
                    time=("time", _from_chunked_array(table["_nano_ts"]).astype("datetime64[ns]")),
                    id=("time", _from_chunked_array(table["id"])),
                ),
            )
        ).groupby("id")
    )
