import base64
import uuid
import typing
from functools import partial
import numpy as np
import pymongo
from dataclasses import dataclass, field, fields, MISSING
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Optional, Union
from collections.abc import Callable
import datetime
from numbers import Real


class MetaID(uuid.UUID):
    @classmethod
    def from_stripped_base64(cls, s: str) -> "MetaID":
        return cls(bytes=base64.urlsafe_b64decode(s + "=" * (4 - len(s) % 4)))

    def to_stripped_base64(self) -> str:
        return base64.urlsafe_b64encode(self.bytes).rstrip(b"=").decode("utf8")

    def __str__(self) -> str:
        return self.to_stripped_base64()


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
    meta: MetaID = _from_query("meta_id", MetaID.from_stripped_base64)
    measurement: str = _from_query("measurement_id", str)
    subject: str = _from_query("subject_id", str)

    group: Optional[str] = _from_query("group_id", str, default=None)
    note: Optional[MetaID] = _from_query("note_id", MetaID.from_stripped_base64, default=None)
    postfix: Optional[str] = _from_query("postfix_id", str, default=None)
    session: Optional[str] = _from_query("session_id", str, default=None)
    study: Optional[str] = _from_query("study_id", str, default=None)
    mongo: Any = _from_query("_id", default=None)
    file_prefix: Optional[PurePath] = _from_query("file_prefix", PureWindowsPath, default=None)

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
        qfields = {k: f(qv) for k, f, qv in ((*converters[qk], qv) for qk, qv in query.items())}
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
class NotesMeta(Meta):
    duration: Optional[int] = _from_query("duration", lambda v: datetime.timedelta(seconds=_to_real(v)), default=None)
    is_fastrak: bool = _from_query("isFasTrak", bool, default=False)
    is_finapres: bool = _from_query("isFinapres", bool, default=False)
    is_metaox: bool = _from_query("isMetaOx", bool, default=False)
    is_pm: bool = _from_query("isPM", bool, default=False)
    is_with_hairline: bool = _from_query("isWithHairline", bool, default=False)
    is_with_probe: bool = _from_query("isWithProbe", bool, default=False)
    measurement_sites: "tuple[str, ...]" = _from_query("measurementSites", tuple, default_factory=tuple)
    pm_info: "dict[str, Any]" = _from_query("pm", default_factory=dict)
    locations_measured: "tuple[str, ...]" = _from_query("locations_measured", tuple, default_factory=tuple)


@dataclass(frozen=True)
class FastrakMeta(Meta):
    is_cm: bool = _from_query("is_fastrak_cm", bool, default=False)


@dataclass(frozen=True)
class NIRSMeta(Meta):
    nirs_distances: "tuple[Real, ...]" = _from_query("nirsDistances", tuple)
    nirs_wavelengths: "tuple[Real, ...]" = _from_query("nirsWavelengths", tuple)
    nirs_hz: Real = _from_query("nirs_hz", _to_real)
    dcs_distances: "tuple[Real, ...]" = _from_query("dcsDistances", tuple)
    dcs_wavelength: Real = _from_query("dcsWavelength", _to_real)
    dcs_hz: Real = _from_query("dcs_hz", _to_real)
    gains: "tuple[Real, ...]" = _from_query("gains", tuple)
    duration: Optional[datetime.timedelta] = _from_query(
        "duration", lambda v: datetime.timedelta(seconds=_to_real(v)), default=None
    )
    nirs_start: Optional[np.datetime64] = _from_query("nirsStartNanoTS", lambda v: np.datetime64(v, "ns"), default=None)
    nirs_end: Optional[np.datetime64] = _from_query("nirsEndNanoTS", lambda v: np.datetime64(v, "ns"), default=None)
    dcs_start: Optional[np.datetime64] = _from_query("dcsStartNanoTS", lambda v: np.datetime64(v, "ns"), default=None)
    dcs_end: Optional[np.datetime64] = _from_query("dcsEndNanoTS", lambda v: np.datetime64(v, "ns"), default=None)


@dataclass(frozen=True)
class FinapresMeta(Meta):
    # device_id
    pass


@dataclass(frozen=True)
class PatientMonitorMeta(Meta):
    pass


create_mongo_client = partial(
    pymongo.MongoClient,
    host="mongos.mongo.svc.cluster.local",
    port=27017,
    ssl=True,
    authSource="$external",
    authMechanism="MONGODB-X509",
    # Changed in version 3.12: ssl_certfile and ssl_keyfile were deprecated in favor of tlsCertificateKeyFile.
    ssl_certfile="/etc/mongo/jhub-keypem.pem",
    ssl_ca_certs="/etc/mongo/root-ca.pem",
)
create_client = create_mongo_client

META_DATABASE_KEY: str = "meta"
META_COLLECTION_KEY: str = "meta3"


def query_meta(client: pymongo.MongoClient, query: dict, fields=None, find_kwargs=None):
    db: pymongo.Database = client[META_DATABASE_KEY]
    col: pymongo.Collection = db[META_COLLECTION_KEY]
    cursor: pymongo.Cursor = col.find(
        filter=query,
        projection=[f if f in ("_id", "meta_id") else f"{f}.val" for f in fields] if fields is not None else None,
        **(find_kwargs or {}),
    )
    yield from ({k: (v["val"] if isinstance(v, dict) and "val" in v else v) for k, v in doc.items()} for doc in cursor)


def query_meta_typed(
    client: pymongo.MongoClient,
    query={},
    find_kwargs=None,
    meta_type: typing.Type[Meta] = Meta,
):
    yield from map(
        meta_type.from_query,
        query_meta(
            client,
            query,
            fields=meta_type.query_keys(),
            find_kwargs=find_kwargs,
        ),
    )


def query_notes_meta(client: pymongo.MongoClient, query={}, find_kwargs=None):
    yield from query_meta_typed(
        client,
        query={"the_type.val": "notes", **query},
        find_kwargs=find_kwargs,
        meta_type=NotesMeta,
    )


def query_fastrak_meta(client: pymongo.MongoClient, query={}, find_kwargs=None):
    yield from query_meta_typed(
        client,
        query={"n_fastrak_dedup.val": {"$exists": True}, **query},
        find_kwargs=find_kwargs,
        meta_type=FastrakMeta,
    )


def query_nirs_meta(client: pymongo.MongoClient, query={}, find_kwargs=None):
    yield from query_meta_typed(
        client,
        query={"n_nirs_dedup.val": {"$exists": True}, **query},
        find_kwargs=find_kwargs,
        meta_type=NIRSMeta,
    )


def query_finapres_meta(client: pymongo.MongoClient, query={}, find_kwargs=None):
    yield from query_meta_typed(
        client,
        query={"n_waveform_dedup.val": {"$exists": True}, **query},
        find_kwargs=find_kwargs,
        meta_type=FinapresMeta,
    )


def query_patient_monitor_meta(client: pymongo.MongoClient, query={}, find_kwargs=None):
    yield from query_meta_typed(
        client,
        query={
            "n_waves_dedup.val": {"$exists": True},
            "n_numerics_dedup.val": {"$exists": True},
            **query,
        },
        find_kwargs=find_kwargs,
        meta_type=PatientMonitorMeta,
    )
