import datetime
import warnings
from functools import partial
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.dataset as pds
import xarray as xr
from fsspec import AbstractFileSystem

from .nirscloud_data import add_meta_coords, nirs_ds_from_table
from .nirscloud_hdfs import HDFS_PREFIX_DEDUP, HDFS_PREFIX_KAFKA_TOPICS, KAFKA_TOPICS_N
from .nirscloud_mongo import Meta, NIRSMeta
from .nirscloud_raw import read_nirsraw

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


read_pq_dataset = partial(
    pds.dataset,
    format="parquet",
    ignore_prefixes=["."],
)


def read_pq_table(fs: AbstractFileSystem, dir_path: str | PurePosixPath) -> pa.Table:
    dir_path = str(dir_path)
    ## NOTE: can't use `fs.isfile` because it won't raise FileNotFoundError
    status = fs.info(dir_path)
    pq_ds = read_pq_dataset(dir_path, filesystem=fs, partitioning=None if status["type"] == "file" else "hive")
    return pq_ds.to_table()


def try_read_pq_table(fs: AbstractFileSystem, *dir_paths: str | PurePosixPath):
    incomplete_dir_paths = []
    for dir_path in dir_paths:
        dir_path = str(dir_path)
        if not fs.exists(dir_path):
            continue
        if has_missing_data(fs, dir_path):
            ## NOTE: check if it has any data before adding it
            if fs.find(dir_path):
                incomplete_dir_paths.append(dir_path)
            continue
        return read_pq_table(fs, dir_path), False
    return [read_pq_table(fs, dir_path) for dir_path in incomplete_dir_paths], True


def try_read_pq_table_from_meta(fs: AbstractFileSystem, meta: Meta, *hdfs_prefix_options: PurePosixPath):
    return try_read_pq_table(fs, *(prefix / meta.hdfs for prefix in hdfs_prefix_options))


def read_nirs_ds_from_meta_raw(meta: NIRSMeta, smb_path: Path):
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

    raw_ds = read_nirsraw(nirsraw_filepath, ndet, nwavelength, nirs_hz=nirs_hz)
    return raw_ds


def try_read_nirs_ds_from_meta_inner(fs: AbstractFileSystem, meta: NIRSMeta, smb_path: Path):
    table, missing = try_read_pq_table_from_meta(
        fs, meta, HDFS_PREFIX_DEDUP / KAFKA_TOPICS_N, HDFS_PREFIX_KAFKA_TOPICS / KAFKA_TOPICS_N
    )
    if not missing:
        raw_ds = nirs_ds_from_table(table, nirs_det_dim="detector")
        ## Need to dedup here, since it may be read from `/kafka/topics/metaox_nirs_rs` which has duplicates still
        return raw_ds.drop_duplicates("time"), False, False
    elif table:
        raw_ds = (
            xr.concat([nirs_ds_from_table(t, nirs_det_dim="detector") for t in table], "time", join="outer")
            .sortby("time")
            .drop_duplicates("time")
        )
        raw_n_time = raw_ds.sizes["time"]
        if raw_n_time >= meta.n_nirs:
            if raw_n_time > meta.n_nirs:
                warnings.warn(f"{meta.meta!r}: Expected {meta.n_nirs} time points, but found {raw_n_time}")
            return raw_ds, False, False
    elif meta.nirsraw_filepath is None:
        raise FileNotFoundError(meta.hdfs)
    else:
        raw_ds = None

    if meta.nirsraw_filepath is None:
        return raw_ds, True, False
    elif meta.nirsraw_filepath.parts[:2] != ("/", "smb"):
        warnings.warn(f'{meta.meta!r}: nirsraw_filepath {meta.nirsraw_filepath} doesn\'t start with "/smb"')
        return raw_ds, True, False
    nirsraw_ds = read_nirs_ds_from_meta_raw(meta, smb_path)
    if raw_ds is None:
        return nirsraw_ds, False, True
    ## NOTE: drop duplicates before sorting to keep the better numerical values
    ds = xr.concat([raw_ds, nirsraw_ds], "time", join="outer").drop_duplicates("time", keep="first").sortby("time")
    return ds, False, True


def try_read_nirs_ds_from_meta(fs: AbstractFileSystem, meta: NIRSMeta, smb_path: Path):
    """Read NIRS data from the cluster using the redudant non-deduplicated data and falling back to
    the `.nirsraw` backup on the fileshare to account for partial data after the
    cluster data loss incident on March 27th 2024

    Parameters
    ----------
    fs
        A wrapped `HdfsFileSystem` to access the cluster
    meta
        A mongo document describing the metadata of a measurement
    smb_path
        The nirsraw filepath in meta always starts with '/smb/', replace it with `smb_path` pointing towards
        the mounted location of the fileshare. On our jupyterhub that is '/home'
    """
    ds, missing, from_nirsraw = try_read_nirs_ds_from_meta_inner(fs, meta, smb_path)
    if missing:
        warnings.warn(f"{meta.meta!r}: missing data")
    if from_nirsraw:
        warnings.warn(f"{meta.meta!r}: using truncated .nirsraw data in place of missing data")
    return add_meta_coords(ds, meta, nirs_det_dim="detector", metaox_to_rel_time=False)
