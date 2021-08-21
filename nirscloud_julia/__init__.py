from .dual_quaternion import DualQuaternion
from .mpl_marker import centered_text_as_path, marker_with_text
from .histos import mpl_histos, mpl_histos_legend
from .nirscloud_mongo import create_client, query_meta, query_fastrak_meta, query_nirs_meta
from .nirscloud_hdfs import create_webhfs_client, read_data_from_meta, read_fastrak_ds_from_meta, read_nirs_ds_from_meta
import xarray as xr

def combine_measurements_ds(fastrak_ds: xr.Dataset, nirs_ds: xr.Dataset) -> xr.Dataset:
    fastrak_measurements = []
    for dt in xr.concat((nirs_ds.nirs_start_time, nirs_ds.nirs_start_time + nirs_ds.duration), dim="t").transpose("measurement", "t"):
        m = str(dt.measurement.values)
        start, end = dt.values
        t_slice = slice(start, end + 1)
        fastrak_time_sliced = fastrak_ds.sel(time=t_slice).drop_vars("measurement").rename_dims(time="fastrak_time") #.rename(time="fastrak_time")
        fastrak_time_sliced.coords["fastrak_time"] = fastrak_time_sliced["time"] - start
        fastrak_time_sliced.coords["measurement"] = m
        fastrak_measurements.append(fastrak_time_sliced.drop_vars("time"))
    measurements = xr.merge((xr.concat(fastrak_measurements, dim="measurement").drop_vars(["meta_id", "session"]), nirs_ds.drop_vars(["meta_id", "session"])))
    measurements.coords["measurement_location"], measurements.coords["measurement_trial"] = measurements.measurement.str.split("split_axis", "_", 1).transpose("split_axis", ...)
    return measurements
