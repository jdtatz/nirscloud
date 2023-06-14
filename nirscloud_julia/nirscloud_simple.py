import importlib.resources
from functools import lru_cache
from json import load
from pathlib import PurePosixPath

import httpx
import numpy as np
import xarray as xr

from .nirscloud_data import (
    add_meta_coords,
    dcs_ds_from_table,
    fastrak_ds_from_table,
    finapres_ds_from_table,
    id_val_dict_from_table,
    id_val_ds_from_table,
    nirs_ds_from_table,
    nk_dict_to_ds,
    nk_dict_to_ecg_da,
    patient_monitor_da_dict_from_table,
    vent_ds_from_table,
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
    NIRSMeta,
    NkAlertMeta,
    NkWaveMeta,
    PatientMonitorMeta,
    VentAlarmsMeta,
    VentMeta,
    VentNumericMeta,
    VentSettingsMeta,
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


def read_vent_table_from_meta(client: WebHDFS[httpx.Client], meta: VentMeta):
    if meta.agg_by_hr:
        return read_table_from_meta(client, meta, HDFS_PREFIX_AGG_HR3, HDFS_PREFIX_AGG_HR2, HDFS_PREFIX_AGG_HR)
    elif meta.agg_by_day:
        return read_table_from_meta(client, meta, HDFS_PREFIX_AGG_DAY)
    else:
        return read_table_from_meta(client, meta, HDFS_PREFIX_KAFKA_TOPICS)


def read_nk_waves_ds_from_meta(client: WebHDFS[httpx.Client], meta: NkWaveMeta):
    table = read_vent_table_from_meta(client, meta)
    nk_wave_dict = id_val_dict_from_table(table)
    ecg_da = nk_dict_to_ecg_da(nk_wave_dict)
    nk_ds = nk_dict_to_ds(nk_wave_dict)
    return ecg_da, nk_ds


def read_vent_waves_ds_from_meta(client: WebHDFS[httpx.Client], meta: VentWaveMeta):
    table = read_vent_table_from_meta(client, meta)
    return vent_ds_from_table(table)


def read_alarms_ds_from_meta(client: WebHDFS[httpx.Client], meta: VentAlarmsMeta):
    table = read_vent_table_from_meta(client, meta)
    return vent_ds_from_table(table)


@lru_cache
def _numerics_map() -> dict[str, str]:
    resources = importlib.resources.files("nirscloud_julia")
    with resources.joinpath("numerics_descr.json").open("r") as f:
        return load(f)


@lru_cache
def _settings_map() -> dict[str, str]:
    resources = importlib.resources.files("nirscloud_julia")
    with resources.joinpath("settings_descr.json").open("r") as f:
        return load(f)


@lru_cache
def _alarm_desc_mapping() -> dict[str, str]:
    resources = importlib.resources.files("nirscloud_julia")
    with resources.joinpath("alarm_settings_descr.json").open("r") as f:
        return load(f)


def read_vent_numerics_ds_from_meta(client: WebHDFS[httpx.Client], meta: VentNumericMeta):
    table = read_vent_table_from_meta(client, meta)
    vent_n_ds = id_val_ds_from_table(table)
    for k, s_attrs in _numerics_map().items():
        if k in vent_n_ds.variables:
            vent_n_ds[k].attrs.update(s_attrs)
    return vent_n_ds


# FIXME: Emit a warning if there is somehow another kind of setting beside vent or alarm
def read_settings_ds_from_meta(client: WebHDFS[httpx.Client], meta: VentSettingsMeta):
    table = read_vent_table_from_meta(client, meta)
    settings = id_val_dict_from_table(table)
    vent_settings_ds = xr.Dataset({k: da for k, da in settings.items() if k.startswith("s")})
    for k, s_attrs in _settings_map().items():
        if k in vent_settings_ds.variables:
            vent_settings_ds[k].attrs.update(s_attrs)
    alarm_settings_ds = xr.Dataset({k: da for k, da in settings.items() if k.startswith("a")})
    for k, s_attrs in _alarm_desc_mapping().items():
        if k in alarm_settings_ds.variables:
            alarm_settings_ds[k].attrs.update(s_attrs)
    return vent_settings_ds, alarm_settings_ds
