import datetime
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import xarray as xr
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from metaox_parser import read_dcsraw, read_nirsraw

from .constants import (
    KAFKA_TOPICS_D,
    KAFKA_TOPICS_N,
)
from .data import (
    dcs_ds_from_table,
    fastrak_raw_stacked_ds_from_table,
    nirs_ds_from_table,
)
from .metadata import convert_dcs_meta, convert_fastrak_meta, convert_nirs_meta, try_pop_extra_attrs, update_metadata
from .mongo import DCSMeta, FastrakMeta, Meta, NIRSMeta

__all__ = ["try_read_dcs_ds_from_meta", "try_read_fastrak_raw_stacked_ds_from_meta", "try_read_nirs_ds_from_meta"]

## After NE136 on 2024-03-27 '/nirscloud/dedup/metaox_nirs_rs/_study_id=CCHU/_group_id=_/_subject_id=NE136/_the_date=2024-03-27/_meta_id=xNR9hfs21EKC2dk782WfTA'
missing_start_date = datetime.datetime(2024, 3, 27, 11, 16, tzinfo=datetime.UTC)
## TODO: more precise
missing_end_date = datetime.datetime(2024, 3, 30, tzinfo=datetime.UTC)


## NOTE: this won't work for patient monitor data
def has_missing_data(fs: AbstractFileSystem, dir_path: str | PurePosixPath):
    for root, _dirs, _files in fs.walk(str(dir_path)):
        mt = fs.modified(root)
        if mt.tzinfo is None or mt.tzinfo.utcoffset(mt) is None:
            mt = mt.replace(tzinfo=datetime.UTC)
        if missing_start_date < mt < missing_end_date:
            return True
    return False


pq_dataset = partial(
    pds.dataset,
    format=pds.ParquetFileFormat(read_options={"list_type": pa.LargeListType}),
    ignore_prefixes=["."],
)


def try_read_pq_dataset_timestamp_sorted(
    fs: AbstractFileSystem, dir_path: str | PurePosixPath
) -> pds.FileSystemDataset | None:
    dir_path = str(dir_path)
    if not fs.exists(dir_path):
        return None
    infos = list(fs.ls(dir_path, detail=True))
    found = []
    for info in infos:
        fname = info["name"]
        if info["type"] == "file":
            # TODO: use the standard key once they decide on one https://github.com/fsspec/filesystem_spec/issues/526
            if "mtime" in info:
                found.append((info["mtime"], fname))
            elif "modified" in info:
                found.append((info["modified"], fname))
            elif "modification_time" in info:
                found.append((info["modification_time"], fname))
            else:
                found.append((fs.modified(fname), fname))
        elif not fname.startswith("."):
            # TODO: better message and more accurate exception type
            msg = f"Unexpected directory '{fname}' found in parquet dataset folder"
            raise ValueError(msg)
    _, pq_paths = zip(*sorted(found), strict=True)
    if not pq_paths:
        return None
    return pq_dataset(pq_paths, filesystem=fs, partitioning=None)


def try_read_pq_dataset_from_parts(
    fs: AbstractFileSystem, dir_path: str | PurePosixPath, missing: set[int]
) -> tuple[pds.FileSystemDataset | None, set[int]]:
    dir_path = str(dir_path)
    missing = missing.copy()
    pq_paths = {}
    for root, _dirs, files in fs.walk(dir_path):
        if not files:
            continue
        part_dir = PurePosixPath(root)
        assert part_dir.stem.startswith("_part=")
        part = int(part_dir.stem.removeprefix("_part="))
        if part not in missing:
            continue
        missing.remove(part)
        assert len(files) == 1
        (pq_file,) = files
        pq_paths[part] = str(part_dir / pq_file)
    sorted_pq_paths = [f for _p, f in sorted(pq_paths.items())]
    if not sorted_pq_paths:
        return None, missing
    return pq_dataset(sorted_pq_paths, filesystem=fs, partitioning="hive", partition_base_dir=dir_path), missing


_HDFS_ROOT = PurePosixPath("/")


def _get_fs_hdfs_root(fs: Path | AbstractFileSystem) -> tuple[AbstractFileSystem, Path | PurePosixPath]:
    if isinstance(fs, AbstractFileSystem):
        return fs, _HDFS_ROOT
    else:
        if not isinstance(fs, Path):
            raise NotImplementedError
        return LocalFileSystem(), fs


def try_read_pq_dataset_from_meta_parts(
    hdfs: Path | AbstractFileSystem,
    meta: Meta,
    min_part: int | None,
    max_part: int | None,
    kafka_topic: str,
    agg_prefix: Literal["agg", "agg_by_hr", "agg_by_hr3", "agg_by_day", "ts_by_hr3"] | None = None,
):
    fs, hdfs_root = _get_fs_hdfs_root(hdfs)
    if agg_prefix is not None:
        agg_dir_path = hdfs_root / "nirscloud" / agg_prefix / kafka_topic / meta.hdfs
        # TODO: Ensure there is only a single parquet file
        agg_pq_ds = try_read_pq_dataset_timestamp_sorted(fs, agg_dir_path)
        if agg_pq_ds is not None:
            return agg_pq_ds, None, False
    dedup_dir_path = hdfs_root / "nirscloud/dedup" / kafka_topic / meta.hdfs
    raw_dir_path = hdfs_root / "kafka/topics" / kafka_topic / meta.hdfs
    if not (fs.exists(str(dedup_dir_path)) or fs.exists(str(raw_dir_path))):
        msg = f"{meta.hdfs} not found in kafka topic {kafka_topic!r}"
        raise FileNotFoundError(msg)
    assert (min_part is None) == (max_part is None)
    dedup_pq_ds = None
    filter_expr = None
    if min_part is not None and max_part is not None:
        missing = set(range(min_part, 1 + max_part))
        dedup_pq_ds, missing = try_read_pq_dataset_from_parts(fs, dedup_dir_path, missing)
        if dedup_pq_ds is not None:
            if len(missing) == 0:
                return dedup_pq_ds, None, False
            filter_expr = pc.field("_part").isin(missing)
    raw_pq_ds = try_read_pq_dataset_timestamp_sorted(fs, raw_dir_path)
    if raw_pq_ds is None:
        return dedup_pq_ds, None, True
    # TODO: is this really reliable?
    if not has_missing_data(fs, raw_dir_path):
        return None, raw_pq_ds, False
    if filter_expr is not None:
        raw_pq_ds = raw_pq_ds.filter(filter_expr)
    # TOOD: return a single `UnionDataset` once it's supported, current error is `ValueError: Creating an UnionDataset from filtered or projected Datasets is currently not supported`
    return dedup_pq_ds, raw_pq_ds, True


def try_read_raw_ds_from_meta_parts(
    from_table: Callable[[pa.RecordBatch], xr.Dataset],
    hdfs: Path | AbstractFileSystem,
    meta: Meta,
    expected_n: int,
    min_part: int | None,
    max_part: int | None,
    *,
    kafka_topic: str,
    agg_prefix: PurePosixPath | None = None,
    dim: str | tuple[str, list[str]],
    cols: list[str] | None = None,
):
    dedup_pq_ds, raw_pq_ds, missing = try_read_pq_dataset_from_meta_parts(
        hdfs, meta, min_part, max_part, kafka_topic, agg_prefix
    )
    needs_dedup = raw_pq_ds is not None
    pq_dss = []
    if dedup_pq_ds is not None:
        pq_dss.append(dedup_pq_ds)
    if raw_pq_ds is not None:
        pq_dss.append(raw_pq_ds)

    if pq_dss:
        if not isinstance(dim, str):
            dim, stacked = dim
            indexes = {dim: stacked}
            sort_dim = list(stacked)
        else:
            indexes = None
            sort_dim = dim
        raw_ds = xr.concat(
            [from_table(rb) for pq_ds in pq_dss for rb in pq_ds.to_batches(columns=cols) if rb.num_rows > 0], dim=dim
        ).sortby(sort_dim)
        if needs_dedup:
            n_dup = raw_ds.sizes[dim]
            if indexes is not None:
                # TODO: do this without a multi-index
                raw_ds = raw_ds.set_index(indexes).drop_duplicates(dim).reset_index(dim)
            else:
                raw_ds = raw_ds.drop_duplicates(dim)
            n_dedup = raw_ds.sizes[dim]
            if n_dedup == n_dup:
                warnings.warn("de-dup wasn't needed")
        # assert not np.any(np.diff(raw_ds[dim]) == 0)
        raw_n_time = raw_ds.sizes[dim]
        if not missing:
            return raw_ds, False
        elif raw_n_time >= expected_n:
            if raw_n_time > expected_n:
                warnings.warn(f"{meta.meta!r}: Expected {expected_n} points, but found {raw_n_time}")
            return raw_ds, False
    else:
        raw_ds = None
    return raw_ds, True


def read_nirs_ds_from_meta_raw(
    meta: NIRSMeta,
    smb_path: Path,
    *,
    contiguous: bool = True,
    transpose: bool = True,
):
    if meta.nirsraw_filepath is None:
        raise ValueError("`meta.nirsraw_filepath` is `None`")
    elif meta.nirsraw_filepath.parts[:2] != ("/", "smb"):
        raise ValueError(f'`meta.nirsraw_filepath`({meta.nirsraw_filepath}) doesn\'t start with "/smb/"')
    nirsraw_filepath = smb_path.joinpath(*meta.nirsraw_filepath.parts[2:])
    if meta.nirs_distances is None:
        warnings.warn(f"{meta.meta!r}: Missing `nirs_distances`, assuming 4 detectors")
        ndet = 4
    else:
        ndet = len(meta.nirs_distances)
    nwavelength = 0 if meta.nirs_wavelengths is None else len(meta.nirs_wavelengths)

    if meta.nirs_hz is None:
        warnings.warn(f"{meta.meta!r}: Missing `meta.nirs_hz`, assuming 20 Hz")
        nirs_hz = 20
    elif not meta.nirs_hz.is_integer():
        raise ValueError(f"`meta.nirs_hz` {meta.nirs_hz} is not an integer")
    else:
        nirs_hz = int(meta.nirs_hz)

    return read_nirsraw(
        nirsraw_filepath,
        ndet,
        nwavelength,
        nirs_hz=nirs_hz,
        contiguous=contiguous,
        transpose=transpose,
    )


def read_dcs_ds_from_meta_raw(
    meta: DCSMeta,
    smb_path: Path,
    *,
    contiguous: bool = True,
    transpose: bool = True,
):
    if meta.dcsraw_filepath is None:
        raise ValueError("`meta.dcsraw_filepath` is `None`")
    elif meta.dcsraw_filepath.parts[:2] != ("/", "smb"):
        raise ValueError(f'`meta.dcsraw_filepath`({meta.dcsraw_filepath}) doesn\'t start with "/smb/"')
    dcsraw_filepath = smb_path.joinpath(*meta.dcsraw_filepath.parts[2:])

    if meta.dcs_hz is None:
        warnings.warn(f"{meta.meta!r}: Missing `meta.dcs_hz`, assuming 50 Hz")
        dcs_hz = 50
    elif not meta.dcs_hz.is_integer():
        raise ValueError(f"`meta.dcs_hz` {meta.dcs_hz} is not an integer")
    else:
        dcs_hz = int(meta.dcs_hz)

    return read_dcsraw(
        dcsraw_filepath,
        dcs_hz=dcs_hz,
        flipped_banks=meta.flipped_banks,
        contiguous=contiguous,
        transpose=transpose,
    )


def try_read_nirs_ds_from_meta_inner(
    hdfs: Path | AbstractFileSystem,
    meta: NIRSMeta,
    smb_path: Path,
    *,
    prefer_timedelta: bool = False,
    transpose: bool = True,
):
    from_table = partial(nirs_ds_from_table, prefer_timedelta=prefer_timedelta, transpose=transpose, sort=False)
    raw_ds, missing = try_read_raw_ds_from_meta_parts(
        from_table,
        hdfs,
        meta,
        meta.n_nirs,
        meta.n_nirs_min_part,
        meta.n_nirs_max_part,
        kafka_topic=KAFKA_TOPICS_N,
        dim="time",
        cols=["_nano_ts", "_offset_nano_ts", "ac", "phase", "dc", "dark", "aux"],
    )
    if not missing:
        return raw_ds, False, False
    elif meta.nirsraw_filepath is None:
        raise FileNotFoundError(meta.hdfs)

    if meta.nirsraw_filepath is None:
        return raw_ds, True, False
    elif meta.nirsraw_filepath.parts[:2] != ("/", "smb"):
        warnings.warn(f'{meta.meta!r}: nirsraw_filepath {meta.nirsraw_filepath} doesn\'t start with "/smb"')
        return raw_ds, True, False
    nirsraw_ds = read_nirs_ds_from_meta_raw(meta, smb_path, transpose=transpose, contiguous=raw_ds is None)
    if not prefer_timedelta:
        if raw_ds and "start" in raw_ds.attrs:
            nirsraw_ds["time"] = raw_ds.attrs["start"] + nirsraw_ds["time"]
        elif meta.nirs_start is not None:
            nirsraw_ds["time"] = meta.nirs_start + nirsraw_ds["time"]
    if raw_ds is None:
        return nirsraw_ds, False, True
    ## NOTE: drop duplicates before sorting to keep the better numerical values
    ds = xr.concat([raw_ds, nirsraw_ds], "time", join="outer").drop_duplicates("time", keep="first").sortby("time")
    return ds, False, True


def try_read_nirs_ds_from_meta(
    hdfs: Path | AbstractFileSystem,
    meta: NIRSMeta,
    smb_path: Path,
    *,
    prefer_timedelta: bool = False,
    transpose: bool = True,
):
    """Read NIRS data from the cluster using the redudant non-deduplicated data and falling back to
    the `.nirsraw` backup on the fileshare to account for partial data after the
    cluster data loss incident on March 27th 2024

    Parameters
    ----------
    hdfs
        The mounted location or `fsspec` file-system interface to access the hdfs data
    meta
        A mongo document describing the metadata of a measurement
    smb_path
        The nirsraw filepath in meta always starts with '/smb/', replace it with `smb_path` pointing towards
        the mounted location of the fileshare. On our jupyterhub that is '/home'
    prefer_timedelta
        prefer the `time` coordinate as a timedelta64 instead of datetime64
    transpose
        transpose the `time` coordinate to be the last coordinate instead of the first
    """
    ds, missing, from_nirsraw = try_read_nirs_ds_from_meta_inner(
        hdfs, meta, smb_path, prefer_timedelta=prefer_timedelta, transpose=transpose
    )
    if missing:
        warnings.warn(f"{meta.meta!r}: missing data")
    if from_nirsraw:
        warnings.warn(f"{meta.meta!r}: using truncated .nirsraw data in place of missing data")
    metadata = convert_nirs_meta(meta)
    return try_pop_extra_attrs(update_metadata(ds, metadata))


def try_read_dcs_ds_from_meta_inner(
    hdfs: Path | AbstractFileSystem,
    meta: DCSMeta,
    smb_path: Path,
    *,
    prefer_timedelta: bool = False,
    transpose: bool = True,
):
    from_table = partial(
        dcs_ds_from_table,
        flipped_banks=meta.flipped_banks,
        prefer_timedelta=prefer_timedelta,
        transpose=transpose,
        sort=False,
    )
    raw_ds, missing = try_read_raw_ds_from_meta_parts(
        from_table,
        hdfs,
        meta,
        meta.n_dcs,
        meta.n_dcs_min_part,
        meta.n_dcs_max_part,
        kafka_topic=KAFKA_TOPICS_D,
        dim="time",
        cols=["_nano_ts", "_offset_nano_ts", "t", "CPS", "t_val"],
    )
    if not missing:
        return raw_ds, False, False
    elif meta.dcsraw_filepath is None:
        raise FileNotFoundError(meta.hdfs)

    if meta.dcsraw_filepath is None:
        return raw_ds, True, False
    elif meta.dcsraw_filepath.parts[:2] != ("/", "smb"):
        warnings.warn(f'{meta.meta!r}: dcsraw_filepath {meta.dcsraw_filepath} doesn\'t start with "/smb"')
        return raw_ds, True, False
    dcsraw_ds = read_dcs_ds_from_meta_raw(meta, smb_path, transpose=transpose, contiguous=raw_ds is None)
    if not prefer_timedelta:
        if raw_ds and "start" in raw_ds.attrs:
            dcsraw_ds["time"] = raw_ds.attrs["start"] + dcsraw_ds["time"]
        elif meta.dcs_start is not None:
            dcsraw_ds["time"] = meta.dcs_start + dcsraw_ds["time"]
    if raw_ds is None:
        return dcsraw_ds, False, True
    ## NOTE: drop duplicates before sorting to keep the better numerical values
    ds = xr.concat([raw_ds, dcsraw_ds], "time", join="outer").drop_duplicates("time", keep="first").sortby("time")
    return ds, False, True


def try_read_dcs_ds_from_meta(
    hdfs: Path | AbstractFileSystem,
    meta: DCSMeta,
    smb_path: Path,
    *,
    prefer_timedelta: bool = False,
    transpose: bool = True,
):
    """Read DCS data from the cluster using the redudant non-deduplicated data and falling back to
    the `.dcsraw` backup on the fileshare to account for partial data after the
    cluster data loss incident on March 27th 2024

    Parameters
    ----------
    hdfs
        The mounted location or `fsspec` file-system interface to access the hdfs data
    meta
        A mongo document describing the metadata of a measurement
    smb_path
        The dcsraw filepath in meta always starts with '/smb/', replace it with `smb_path` pointing towards
        the mounted location of the fileshare. On our jupyterhub that is '/home'
    prefer_timedelta
        prefer the `time` coordinate as a timedelta64 instead of datetime64
    transpose
        transpose the `time` coordinate to be the last coordinate instead of the first
    """
    ds, missing, from_dcsraw = try_read_dcs_ds_from_meta_inner(
        hdfs, meta, smb_path, prefer_timedelta=prefer_timedelta, transpose=transpose
    )
    if missing:
        warnings.warn(f"{meta.meta!r}: missing data")
    if from_dcsraw:
        warnings.warn(f"{meta.meta!r}: using truncated .dcsraw data in place of missing data")
    metadata = convert_dcs_meta(meta)
    return try_pop_extra_attrs(update_metadata(ds, metadata))


def try_read_fastrak_raw_stacked_ds_from_meta_inner(
    hdfs: Path | AbstractFileSystem,
    meta: FastrakMeta,
    *,
    prefer_timedelta: bool = False,
):
    # Any measurement before/on 2021-04-13 is only in "fastrak_s"
    assert meta.date > datetime.date(2021, 4, 13)
    return try_read_raw_ds_from_meta_parts(
        partial(fastrak_raw_stacked_ds_from_table, prefer_timedelta=prefer_timedelta),
        hdfs,
        meta,
        meta.n_fastrak,
        meta.n_fastrak_min_part,
        meta.n_fastrak_max_part,
        # FIXME: there's nothing in the metadata to determine if the data is in "fastrak2_s" or "fastrak_s", but most of the data in "fastrak_s" is also in "fastrak2_s"
        kafka_topic="fastrak_cm_s" if meta.is_cm else "fastrak2_s",
        agg_prefix="agg",
        dim=("stacked", ["idx", "time", "list_id"]),
        # FIXME: data in the "fastrak_s" topic doesn't have the "list_id" variable
        cols=["_nano_ts", "_offset_nano_ts", "idx", "list_id", "x", "y", "z", "a", "e", "r"],
    )


def try_read_fastrak_raw_stacked_ds_from_meta(
    hdfs: Path | AbstractFileSystem,
    meta: FastrakMeta,
    *,
    prefer_timedelta: bool = False,
):
    """Read Fastrak data from the cluster using the redudant non-deduplicated data to account for partial data
    after the cluster data loss incident on March 27th 2024

    Parameters
    ----------
    hdfs
        The mounted location or `fsspec` file-system interface to access the hdfs data
    meta
        A mongo document describing the metadata of a measurement
    prefer_timedelta
        prefer the `time` coordinate as a timedelta64 instead of datetime64
    """
    ds, missing = try_read_fastrak_raw_stacked_ds_from_meta_inner(hdfs, meta, prefer_timedelta=prefer_timedelta)
    if missing:
        warnings.warn(f"{meta.meta!r}: missing data")
    metadata = convert_fastrak_meta(meta)
    return try_pop_extra_attrs(update_metadata(ds, metadata))
