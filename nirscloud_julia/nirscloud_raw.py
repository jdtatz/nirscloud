from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import DTypeLike


def _add_time_coord(ds: xr.Dataset, freq_hz: Optional[int]):
    if freq_hz is None:
        return ds

    ## FIXME: go back to milliseconds by default, once xarray exact alignment works with differing units?
    ## FIXME: swap to microseconds, to support {16, 32, 64, 80} Hz?
    dt, r = divmod(1_000_000_000, freq_hz)
    assert r == 0
    ## TODO: replace with a `pd.RangeIndex`?
    time = np.arange(0, dt * ds.sizes["time"], dt).astype("timedelta64[ns]")
    ds.coords["time"] = ["time"], time
    return ds


def read_nirsraw(
    fname: str | Path,
    ndet: int,
    nwavelength: Optional[int] = None,
    dtype: Optional[DTypeLike] = np.float32,
    *,
    nirs_hz: Optional[int] = None,
    contiguous: bool = True,
):
    nirsraw = np.loadtxt(fname, delimiter="\t", dtype=dtype)
    ntime, ncol = nirsraw.shape
    if ndet < 1:
        raise ValueError("`ndet` must be a positive integer")
    elif nwavelength is None:
        nwavelength = (ncol - 1 - 2 * ndet) // (3 * ndet)

    expected_ncol = 1 + 3 * nwavelength * ndet + 2 * ndet
    if ncol != expected_ncol:
        raise ValueError(f"Expected {expected_ncol} columns, but found {ncol} columns")
    sections = np.add.accumulate([1] + [nwavelength * 3 * ndet] + [ndet] * 2)
    assert sections[-1] == ncol
    _idx, raw_data, aux, dark = np.split(nirsraw.T, sections[:-1], axis=0)
    _idx = _idx.squeeze(axis=0)
    assert np.allclose(np.diff(_idx), 1) and _idx[0] == 0
    ac, dc, phase = raw_data.T.reshape(ntime, ndet, nwavelength, 3).transpose((3, 1, 2, 0))
    ds = xr.Dataset(
        {
            "ac": (("detector", "wavelength", "time"), ac),
            "dc": (("detector", "wavelength", "time"), dc),
            "phase": (("detector", "wavelength", "time"), phase),
            "aux": (("detector", "time"), aux),
            "dark": (("detector", "time"), dark),
        }
    )
    # NOTE: this function is written such that no intermediary arrays are made.
    #       all arrays, including the result, are just views of the original `nirsraw`
    if contiguous:
        # NOTE: because `_idx` was dropped, this should result in less memory used and
        #       contiguous arrays yield better performance
        ds = ds.map(np.ascontiguousarray)
    return _add_time_coord(ds, nirs_hz)


def read_dcsraw(
    fname: str | Path,
    dtype: Optional[DTypeLike] = np.float32,
    *,
    flipped_banks: Optional[bool] = None,
    dcs_hz: Optional[int] = None,
):
    _cps = []
    with open(fname, "r") as f:
        for line in f.readlines():
            if line.startswith("CPS"):
                _, *_r_cps = line.strip().split("\t")
                _cps.append([int(v) for v in _r_cps])
    ## NOTE: create the array as F-contiguous, so the transpose will be C-contiguous
    cps = np.array(_cps, order="F").T
    data = np.loadtxt(fname, delimiter="\t", comments="CPS", dtype=dtype).reshape((cps.shape[1], -1, 1 + cps.shape[0]))
    ## NOTE: move time dim to end to speed-up computation
    data = np.ascontiguousarray(data.T)
    tau_0 = np.ascontiguousarray(data[0, :, 0])
    assert np.all(data[0] == tau_0[:, np.newaxis])
    ds = xr.Dataset(
        {
            "counts": (["channel", "time"], cps, {"units": "Hz"}),
            "value": (["channel", "tau", "time"], data[1:]),
        },
        coords={
            "tau": (["tau"], tau_0, {"units": "s"}),
        },
    )
    if flipped_banks is not None:
        if flipped_banks:
            assert ds.sizes["channel"] == 8
            ds = ds.roll({"channel": 4})
        ## NOTE: the channel coordinate is only added if it's known if the banks were flipped or not
        # ds = ds.assign_coords(channel=(["channel"], np.arange(1, 1 + ds.sizes["channel"])))
        ds = ds.assign_coords(
            xr.Coordinates.from_xindex(xr.indexes.PandasIndex(pd.RangeIndex(1, 1 + ds.sizes["channel"]), "channel"))
        )
    return _add_time_coord(ds, dcs_hz)
