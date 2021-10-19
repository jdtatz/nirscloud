import typing
import numpy as np
import pymongo
from dataclasses import dataclass, field, fields, MISSING
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import NewType, Any, Optional
from collections.abc import Callable
import datetime
from numbers import Real

MetaId = NewType("MetaId", str)


def _id(v):
    return v


def _to_real(v: Any) -> Real:
    return v if isinstance(v, Real) else float(v)


def _from_query(
    key: str,
    converter: "Callable[[Any], Any]" = _id,
    default=MISSING,
    default_factory=MISSING,
    **field_kwargs,
):
    default_factory = (lambda: default) if default is not MISSING else default_factory
    return field(
        metadata=dict(
            query_key=key,
            query_converter=converter,
            query_default_factory=default_factory,
        ),
        **field_kwargs,
    )


def _projection_index(v: dict, projection_key: str):
    for k in projection_key.split("."):
        v = v[k]
    return v


@dataclass(frozen=True)
class Meta:
    hdfs: PurePath = _from_query("hdfs_path", PurePosixPath)
    date: datetime.date = _from_query("the_date", datetime.date.fromisoformat)
    meta: MetaId = _from_query("meta_id", MetaId)
    measurement: MetaId = _from_query("measurement_id", MetaId)
    subject: MetaId = _from_query("subject_id", MetaId)

    group: MetaId = _from_query("group_id", MetaId, default="")
    note: MetaId = _from_query("note_id", MetaId, default="")
    postfix: MetaId = _from_query("postfix_id", MetaId, default="")
    session: MetaId = _from_query("session_id", MetaId, default="")
    study: MetaId = _from_query("study_id", MetaId, default="")
    mongo: Any = _from_query("_id", default=None)
    file_prefix: Optional[PurePath] = _from_query(
        "file_prefix", PureWindowsPath, default=None
    )

    @classmethod
    def query_converters(cls) -> "dict[str, tuple[str, Callable[[Any], Any]]]":
        return {
            f.metadata.get("query_key", f.name): (
                f.name,
                f.metadata.get("query_converter", _id),
            )
            for f in fields(cls)
        }

    @classmethod
    def from_query(cls, query: "dict[str, Any]") -> "Meta":
        converters = cls.query_converters()
        qdefaults = dict(cls.query_defaults())
        qfields = {
            k: f(qv) for k, f, qv in ((*converters[qk], qv) for qk, qv in query.items())
        }
        return cls(**{**qdefaults, **qfields})

    @classmethod
    def query_keys(cls):
        yield from (f.metadata.get("query_key", f.name) for f in fields(cls))

    @classmethod
    def query_defaults(cls):
        for f in fields(cls):
            default_factory = f.metadata.get("query_default_factory", MISSING)
            if default_factory is not MISSING:
                yield (f.name, default_factory())

    @classmethod
    def projection_fields(cls):
        return [k if k in ("_id", "meta_id") else f"{k}.val" for k in cls.query_keys()]


@dataclass(frozen=True)
class FastrakMeta(Meta):
    pass


@dataclass(frozen=True)
class NIRSMeta(Meta):
    nirs_distances: "list[Real]" = _from_query("nirsDistances", list)
    nirs_wavelengths: "list[Real]" = _from_query("nirsWavelengths", list)
    nirs_hz: Real = _from_query("nirs_hz", _to_real)
    dcs_distances: "list[Real]" = _from_query("dcsDistances", list)
    dcs_wavelength: Real = _from_query("dcsWavelength", _to_real)
    dcs_hz: Real = _from_query("dcs_hz", _to_real)
    gains: "list[Real]" = _from_query("gains", list)
    duration: Optional[datetime.timedelta] = _from_query("duration", lambda v: datetime.timedelta(seconds=_to_real(v)), default=None)
    nirs_start: Optional[np.datetime64] = _from_query("nirsStartNanoTS", lambda v: np.datetime64(v, 'ns'), default=None)
    nirs_end: Optional[np.datetime64] = _from_query("nirsEndNanoTS", lambda v: np.datetime64(v, 'ns'), default=None)
    dcs_start: Optional[np.datetime64] = _from_query("dcsStartNanoTS", lambda v: np.datetime64(v, 'ns'), default=None)
    dcs_end: Optional[np.datetime64] = _from_query("dcsEndNanoTS", lambda v: np.datetime64(v, 'ns'), default=None)


@dataclass(frozen=True)
class FinapresMeta(Meta):
    # device_id
    pass


@dataclass(frozen=True)
class PatientMonitorMeta(Meta):
    pass


def create_client(
    host="mongos.mongo.svc.cluster.local",
    port=27017,
    ssl=True,
    authSource="$external",
    authMechanism="MONGODB-X509",
    # Changed in version 3.12: ssl_certfile and ssl_keyfile were deprecated in favor of tlsCertificateKeyFile.
    ssl_certfile="/etc/mongo/jhub-keypem.pem",
    ssl_ca_certs="/etc/mongo/root-ca.pem",
    **extra_kwargs,
) -> pymongo.MongoClient:
    return pymongo.MongoClient(
        host=host,
        port=port,
        ssl=ssl,
        authSource=authSource,
        authMechanism=authMechanism,
        ssl_certfile=ssl_certfile,
        ssl_ca_certs=ssl_ca_certs,
        **extra_kwargs,
    )


META_DATABASE_KEY: str = "meta"
META_COLLECTION_KEY: str = "meta3"


def query_meta(client: pymongo.MongoClient, query: dict, fields=None, find_kwargs=None):
    db: pymongo.Database = client[META_DATABASE_KEY]
    col: pymongo.Collection = db[META_COLLECTION_KEY]
    cursor: pymongo.Cursor = col.find(
        filter=query,
        projection=[f if f in ("_id", "meta_id") else f"{f}.val" for f in fields]
        if fields is not None
        else None,
        **(find_kwargs or {}),
    )
    yield from (
        {
            k: (v["val"] if isinstance(v, dict) and "val" in v else v)
            for k, v in doc.items()
        }
        for doc in cursor
    )


def query_meta_typed(client: pymongo.MongoClient, query={}, find_kwargs=None, meta_type: typing.Type[Meta] = Meta):
    yield from map(
        meta_type.from_query,
        query_meta(
            client,
            query,
            fields=meta_type.query_keys(),
            find_kwargs=find_kwargs,
        ),
    )


def query_fastrak_meta(client: pymongo.MongoClient, query={}, find_kwargs=None):
    yield from query_meta_typed(client, query={"n_fastrak_dedup.val": {"$exists": True}, **query}, find_kwargs=find_kwargs, meta_type=FastrakMeta)


def query_nirs_meta(client: pymongo.MongoClient, query={}, find_kwargs=None):
    yield from query_meta_typed(client, query={"n_nirs_dedup.val": {"$exists": True}, **query}, find_kwargs=find_kwargs, meta_type=NIRSMeta)


def query_finapres_meta(client: pymongo.MongoClient, query={}, find_kwargs=None):
    yield from query_meta_typed(client, query={"n_waveform_dedup.val": {"$exists": True}, **query}, find_kwargs=find_kwargs, meta_type=FinapresMeta)


def query_patient_monitor_meta(client: pymongo.MongoClient, query={}, find_kwargs=None):
    yield from query_meta_typed(client, query={"n_waves_dedup.val": {"$exists": True}, "n_numerics_dedup.val": {"$exists": True}, **query}, find_kwargs=find_kwargs, meta_type=PatientMonitorMeta)
