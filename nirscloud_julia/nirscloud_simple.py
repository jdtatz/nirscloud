from pathlib import PurePosixPath

import httpx
import numpy as np
import xarray as xr

from .nirscloud_data import (
    add_meta_coords,
    dcs_ds_from_table,
    fastrak_ds_from_table,
    finapres_ds_from_table,
    nirs_ds_from_table,
    patient_monitor_da_dict_from_table,
    vent_ds_from_table,
    vent_n_ds_from_table,
)
from .nirscloud_hdfs import (
    HDFS_DCS_PREFIXES,
    HDFS_FASTRAK_PREFIXES,
    HDFS_FINAPRES_PREFIXES,
    HDFS_NIRS_PREFIXES,
    HDFS_PM_N_PREFIXES,
    HDFS_PM_PREFIXES,
    HDFS_PREFIX_AGG_DAY,
    HDFS_PREFIX_AGG_HR,
    HDFS_PREFIX_AGG_HR2,
    HDFS_PREFIX_AGG_HR3,
    HDFS_PREFIX_KAFKA_TOPICS,
    PATIENT_MONITOR_COMPONENT_MAPPING,
    read_table_from_meta,
)
from .nirscloud_mongo import (
    DCSMeta,
    FastrakMeta,
    FinapresMeta,
    Meta,
    NIRSMeta,
    NkWaveMeta,
    PatientMonitorMeta,
    VentMeta,
    VentNumericMeta,
    VentWaveMeta,
)
from .webhdfs import WebHDFS


def read_nirs_ds_from_meta(client: WebHDFS[httpx.Client], meta: NIRSMeta):
    table = read_table_from_meta(client, meta, *HDFS_NIRS_PREFIXES)
    ds = nirs_ds_from_table(table)
    return add_meta_coords(ds, meta)


def read_dcs_ds_from_meta(client: WebHDFS[httpx.Client], meta: DCSMeta):
    table = read_table_from_meta(client, meta, *HDFS_DCS_PREFIXES)
    ds = dcs_ds_from_table(table)
    return add_meta_coords(ds, meta)


def read_fastrak_ds_from_meta(client: WebHDFS[httpx.Client], meta: FastrakMeta):
    table = read_table_from_meta(client, meta, *HDFS_FASTRAK_PREFIXES)
    ds = fastrak_ds_from_table(table)
    return add_meta_coords(ds, meta)


def read_finapres_ds_from_meta(client: WebHDFS[httpx.Client], meta: FinapresMeta):
    *newer_hdfs_finapres_prefixs, HDFS_PREFIX_FP_2, _ = HDFS_FINAPRES_PREFIXES
    # FIXME: I could dedup some work here
    if client.status(HDFS_PREFIX_FP_2 / meta.hdfs, strict=False) is not None:
        ds = finapres_ds_from_table(read_table_from_meta(client, meta, HDFS_PREFIX_FP_2))
        # FIXME: time is stored as 'East Africa Time' (UTC+3) and we need it in UTC
        ds["time"] -= np.timedelta64(3, "h")
    else:
        ds = finapres_ds_from_table(read_table_from_meta(client, meta, *newer_hdfs_finapres_prefixs))
    return add_meta_coords(ds, meta)


def _fix_pm_id(pm_id: str):
    _, comp_hex = pm_id.split("-")
    comp = int(comp_hex, base=16)
    return PATIENT_MONITOR_COMPONENT_MAPPING.get(comp, comp_hex)


def read_patient_monitor_das_from_meta(client: WebHDFS[httpx.Client], meta: PatientMonitorMeta):
    return {
        _fix_pm_id(pm_id): add_meta_coords(ds, meta)
        for pm_id, ds in {
            **patient_monitor_da_dict_from_table(read_table_from_meta(client, meta, *HDFS_PM_PREFIXES)),
            **patient_monitor_da_dict_from_table(
                read_table_from_meta(client, meta, *HDFS_PM_N_PREFIXES),
            ),
        }.items()
    }
