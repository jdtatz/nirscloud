import base64
import datetime
import typing
import uuid
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields
from functools import partial
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import pymongo
import pymongo.collection
import pymongo.cursor
import pymongo.database
from bson import ObjectId
from typing_extensions import Literal, dataclass_transform


class MetaID(uuid.UUID):
    @classmethod
    def from_stripped_base64(cls, s: str) -> "MetaID":
        return cls(bytes=base64.urlsafe_b64decode(s + "=" * (4 - len(s) % 4)))

    def to_stripped_base64(self) -> str:
        return base64.urlsafe_b64encode(self.bytes).rstrip(b"=").decode("utf8")

    def __str__(self) -> str:
        return self.to_stripped_base64()


Real = Union[int, float]


def _to_real(v: Any) -> Real:
    return v if isinstance(v, (int, float)) else float(v)


def _projection_index(v: dict, projection_key: str):
    for k in projection_key.split("."):
        v = v[k]
    return v


_T = TypeVar("_T")


def _id_cast(v: Any) -> _T:
    return v


def query_field(
    key: str,
    converter: "Callable[[Any], _T]" = _id_cast,
    default: "_T | Literal[MISSING]" = MISSING,
    default_factory: "Callable[[], _T] | Literal[MISSING]" = MISSING,
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


@dataclass_transform(field_specifiers=(query_field,), kw_only_default=True)
class MongoMetaBase:
    _database_name: ClassVar[str]
    _collection_name: ClassVar[str]
    _default_query: "ClassVar[dict[str, Any]]"
    _kafka_topics: "ClassVar[list[str]]"
    _extra: "dict[str, Any]"
    # _extra: "dict[str, Any]" = field(init=False, default_factory=dict)

    def __init_subclass__(
        cls,
        database_name: str,
        collection_name: Optional[str] = None,
        default_query: "Optional[dict[str, Any]]" = None,
        kafka_topics: "Optional[list[str]]" = None,
        *args,
        init: bool = True,
        frozen: bool = False,
        eq: bool = True,
        order: bool = True,
        **kwargs,
    ):
        cls._database_name = database_name
        cls._collection_name = f"{database_name}3" if collection_name is None else collection_name
        cls._default_query = dict() if default_query is None else default_query
        cls._kafka_topics = [] if kafka_topics is None else kafka_topics
        super().__init_subclass__(*args, **kwargs)
        dataclass(cls, init=init, frozen=frozen, eq=eq, order=order)

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
    def query_converters(cls) -> "dict[str, tuple[str, Callable[[Any], Any]]]":
        return {
            f.metadata.get("query_key", f.name): (
                f.name,
                f.metadata.get("query_converter", _id_cast),
            )
            for f in fields(cls)
        }

    @classmethod
    def from_query(cls, query: "dict[str, Any]"):
        converters = cls.query_converters()
        qdefaults = dict(cls.query_defaults())
        qfields = dict()
        extra = dict()
        for qk, qv in query.items():
            converter = converters.get(qk, None)
            if converter is None:
                extra[qk] = qv
            else:
                k, f = converter
                qfields[k] = f(qv)
        obj = cls(**{**qdefaults, **qfields})
        obj._extra = extra
        return obj
    
    @property
    def kafka_topics(self) -> "list[str]":
        return self._kafka_topics


class Meta(MongoMetaBase, database_name="meta"):
    mongo: ObjectId = query_field("_id")
    hdfs: PurePath = query_field("hdfs_path", PurePosixPath)
    date: datetime.date = query_field("the_date", datetime.date.fromisoformat)
    meta: MetaID = query_field("meta_id", MetaID.from_stripped_base64)
    measurement: str = query_field("measurement_id", str)
    subject: str = query_field("subject_id", str)

    group: Optional[str] = query_field("group_id", str, default=None)
    note_meta: Optional[MetaID] = query_field("note_id", MetaID.from_stripped_base64, default=None)
    measurement_notes: Optional[str] = query_field("note", str, default=None)
    postfix: Optional[str] = query_field("postfix_id", str, default=None)
    session: Optional[str] = query_field("session_id", str, default=None)
    study: Optional[str] = query_field("study_id", str, default=None)
    device: Optional[str] = query_field("device_id", str, default=None)
    operators: "tuple[str, ...]" = query_field("operators", tuple, default_factory=tuple)
    file_prefix: Optional[PurePath] = query_field("file_prefix", PureWindowsPath, default=None)


# class NotesMeta(Meta, database_name="meta"):
#     duration: Optional[int] = query_field("duration", lambda v: datetime.timedelta(seconds=_to_real(v)), default=None)
#     is_fastrak: bool = query_field("isFasTrak", bool, default=False)
#     is_finapres: bool = query_field("isFinapres", bool, default=False)
#     is_metaox: bool = query_field("isMetaOx", bool, default=False)
#     is_pm: bool = query_field("isPM", bool, default=False)
#     is_with_hairline: bool = query_field("isWithHairline", bool, default=False)
#     is_with_probe: bool = query_field("isWithProbe", bool, default=False)
#     measurement_sites: "tuple[str, ...]" = query_field("measurementSites", tuple, default_factory=tuple)
#     pm_info: "dict[str, Any]" = query_field("pm", default_factory=dict)
#     locations_measured: "tuple[str, ...]" = query_field("locations_measured", tuple, default_factory=tuple)
#     txt_filepath: Optional[PurePosixPath] = query_field("txt_filename", PurePosixPath, default=None)
#     yaml_filepath: Optional[PurePosixPath] = query_field("yaml_filename", PurePosixPath, default=None)


class FastrakMeta(
    Meta,
    database_name="meta",
    default_query={"n_fastrak_dedup.val": {"$exists": True}},
    kafka_topics=["fastrak_cm_s", "fastrak2_s"],
):
    is_cm: bool = query_field("is_fastrak_cm", bool, default=False)


class MetaOxMeta(
    Meta,
    database_name="meta",
    default_query={
        "n_nirs_dedup.val": {"$exists": True},
        "isValid.val": {"$ne": False},
    },
):
    nirs_distances: "tuple[Real, ...]" = query_field("nirsDistances", tuple)
    nirs_wavelengths: "Optional[tuple[Real, ...]]" = query_field("nirsWavelengths", tuple, default=None)
    nirs_hz: Real = query_field("nirs_hz", _to_real)
    dcs_distances: "tuple[Real, ...]" = query_field("dcsDistances", tuple)
    dcs_wavelength: "Optional[Real]" = query_field("dcsWavelength", _to_real, default=None)
    dcs_hz: "Optional[Real]" = query_field("dcs_hz", _to_real, default=None)
    gains: "Optional[tuple[Real, ...]]" = query_field("gains", tuple, default=None)
    is_radian: bool = query_field("is_nirs_radian_single", bool, default=False)
    duration: Optional[datetime.timedelta] = query_field(
        "duration", lambda v: datetime.timedelta(seconds=float(v)), default=None
    )
    nirs_start: Optional[np.datetime64] = query_field("nirsStartNanoTS", lambda v: np.datetime64(v, "ns"), default=None)
    nirs_end: Optional[np.datetime64] = query_field("nirsEndNanoTS", lambda v: np.datetime64(v, "ns"), default=None)
    dcs_start: Optional[np.datetime64] = query_field("dcsStartNanoTS", lambda v: np.datetime64(v, "ns"), default=None)
    dcs_end: Optional[np.datetime64] = query_field("dcsEndNanoTS", lambda v: np.datetime64(v, "ns"), default=None)
    nirsraw_filepath: Optional[PurePosixPath] = query_field("nirsraw_filename", PurePosixPath, default=None)
    dcsraw_filepath: Optional[PurePosixPath] = query_field("dcsraw_filename", PurePosixPath, default=None)


class NIRSMeta(
    MetaOxMeta,
    database_name="meta",
    default_query={
        "n_nirs_dedup.val": {"$exists": True},
        "isValid.val": {"$ne": False},
    },
    kafka_topics=["metaox_nirs_rs", "metaox_nirs_s"],
):
    pass


class DCSMeta(
    MetaOxMeta,
    database_name="meta",
    default_query={
        "n_dcs_dedup.val": {"$exists": True},
        "isValid.val": {"$ne": False},
    },
    kafka_topics=["metaox_dcs_s"],
):
    pass


class FinapresMeta(
    Meta, 
    database_name="meta", 
    default_query={"n_waveform_dedup.val": {"$exists": True}},
    kafka_topics=["finapres_waveform_su", "finapres_waveform2_s", "finapres_waveform_s"],
):
    # device_id
    pass


class PatientMonitorMeta(
    Meta,
    database_name="meta",
    default_query={"n_waves_dedup.val": {"$exists": True}, "n_numerics_dedup.val": {"$exists": True}},
):
    pass


class PatientMonitorWavesMeta(
    Meta,
    database_name="meta",
    default_query={"n_waves_dedup.val": {"$exists": True}},
    kafka_topics=["ixtrend_waves5_s", "ixtrend_waves4_s", "ixtrend_waves3_s", "ixtrend_waves2_s", "ixtrend_waves"],
):
    pass


class PatientMonitorNumericsMeta(
    Meta,
    database_name="meta",
    default_query={"n_numerics_dedup.val": {"$exists": True}},
    kafka_topics=["ixtrend_numerics2_s", "ixtrend_numerics_s", "ixtrend_numerics"],
):
    pass


class VentMeta(MongoMetaBase, database_name="meta_by_bed"):
    mongo: Any = query_field("_id", default=None)
    meta: str = query_field("meta_id")
    bed: str = query_field("bed_id")
    device: str = query_field("device_id")
    topic: str = query_field("the_topic")
    date: datetime.date = query_field("the_date", datetime.date.fromisoformat)
    count: int = query_field("count")
    timestamp: np.datetime64 = query_field("the_nano_ts", lambda v: np.datetime64(v, "ns"))
    hour: Optional[str] = query_field("hr", default=None)
    agg_by_hr: Optional[bool] = query_field("is_agg_by_hr", default=None)
    agg_by_day: Optional[bool] = query_field("is_agg_by_day", default=None)
    removed_kafka_topic: Optional[bool] = query_field("is_remove_kafka_topics", default=None)

    def hdfs_path(self):
        from nirscloud_julia.nirscloud_hdfs import HDFS_PREFIX_AGG_HR, HDFS_PREFIX_AGG_DAY, HDFS_PREFIX_KAFKA_TOPICS

        if self.agg_by_hr:
            prefix = HDFS_PREFIX_AGG_HR
        elif self.agg_by_day:
            prefix = HDFS_PREFIX_AGG_DAY
        else:
            prefix = HDFS_PREFIX_KAFKA_TOPICS
        p = prefix / self.topic / f"_device_id={self.device}" / f"_bed_id={self.bed}" / f"_the_date={self.date}"
        if self.hour is not None:
            p = p / f"_hr={self.hour}"
        return p

    @property
    def kafka_topics(self) -> "list[str]":
        return [self.topic, *super().kafka_topics]


class VentWaveMeta(
    VentMeta,
    database_name="meta_by_bed",
    default_query={"the_topic.val": {"$regex": "bch_vent_waves.*"}},
    kafka_topics=["bch_vent_waves2_s"],
):
    pass


class VentNumericMeta(
    VentMeta,
    database_name="meta_by_bed_day",
    default_query={"the_topic.val": {"$regex": "bch_vent_numerics.*"}},
    kafka_topics=["bch_vent_numerics2_s"],
):
    pass


class VentSettingsMeta(
    VentMeta,
    database_name="meta_by_bed_day",
    default_query={"the_topic.val": {"$regex": "bch_vent_settings.*"}},
    kafka_topics=["bch_vent_settings2_s"],
):
    pass


class VentAlarmsMeta(
    VentMeta,
    database_name="meta_by_bed_day",
    default_query={"the_topic.val": {"$regex": "bch_vent_alarms.*"}},
    kafka_topics=["bch_vent_alarms2_s"],
):
    pass


class NkWaveMeta(
    VentMeta,
    database_name="meta_by_bed",
    default_query={"the_topic.val": {"$regex": "nk_waves.*"}},
    kafka_topics=["nk_waves_NICU"],
):
    pass


class NkAlertMeta(
    VentMeta,
    database_name="meta_by_bed",
    default_query={"the_topic.val": {"$regex": "nk_alert.*"}},
    kafka_topics=["nk_alert_NICU_2"],
):
    pass


create_mongo_client = partial(
    pymongo.MongoClient,
    host="mongos.mongo.svc.cluster.local",
    port=27017,
    ssl=True,
    authSource="$external",
    authMechanism="MONGODB-X509",
    tlsCertificateKeyFile="/etc/mongo/jhub-keypem.pem",
    tlsCAFile="/etc/mongo/root-ca.pem",
)

META_DATABASE_KEY: str = "meta"
META_COLLECTION_KEY: str = "meta3"


def query_meta(
    client: pymongo.MongoClient,
    query: dict,
    fields=None,
    find_kwargs=None,
    database_name=META_DATABASE_KEY,
    collection_name=META_COLLECTION_KEY,
):
    db: pymongo.database.Database = client[database_name]
    col: pymongo.collection.Collection = db[collection_name]
    cursor: pymongo.cursor.Cursor = col.find(
        filter=query,
        projection=[f if f in ("_id", "meta_id") else f"{f}.val" for f in fields] if fields is not None else None,
        **(find_kwargs or {}),
    )
    yield from ({k: (v["val"] if isinstance(v, dict) and "val" in v else v) for k, v in doc.items()} for doc in cursor)


def query_meta_typed(
    client: pymongo.MongoClient,
    query={},
    find_kwargs=None,
    meta_type: typing.Type[MongoMetaBase] = Meta,
):
    yield from map(
        meta_type.from_query,
        query_meta(
            client,
            {**meta_type._default_query, **query},
            # fields=meta_type.query_keys(),
            find_kwargs=find_kwargs,
            database_name=meta_type._database_name,
            collection_name=meta_type._collection_name,
        ),
    )


# query_notes_meta = partial(query_meta_typed, meta_type=NotesMeta)
query_fastrak_meta = partial(query_meta_typed, meta_type=FastrakMeta)
query_nirs_meta = partial(query_meta_typed, meta_type=NIRSMeta)
query_dcs_meta = partial(query_meta_typed, meta_type=DCSMeta)
query_finapres_meta = partial(query_meta_typed, meta_type=FinapresMeta)
query_patient_monitor_meta = partial(query_meta_typed, meta_type=PatientMonitorMeta)
query_vent_meta = partial(query_meta_typed, meta_type=VentMeta)
query_vent_waves_meta = partial(query_meta_typed, meta_type=VentWaveMeta)
query_vent_n_meta = partial(query_meta_typed, meta_type=VentNumericMeta)
query_nk_waves_meta = partial(query_meta_typed, meta_type=NkWaveMeta)
