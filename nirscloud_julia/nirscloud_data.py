from pathlib import PosixPath

import numpy as np
import pandas as pd
import xarray as xr


def read_nirsraw(nirsraw_path: PosixPath, nirsraw_rhos: tuple, nirsraw_wavelengths: tuple):
    if nirsraw_path.parts[:2] == ("/", "smb"):
        nirsraw_path = PosixPath("/", "home", *nirsraw_path.parts[2:])
    with nirsraw_path.open() as f:
        nirsraw = np.genfromtxt(f, delimiter="\t")
    ndet = len(nirsraw_rhos)
    nwave = len(nirsraw_wavelengths)
    _idx, raw_data, aux, dark, _empty = np.split(
        nirsraw, np.add.accumulate([1] + [nwave * 3 * ndet] + [ndet] * 2), axis=1
    )
    assert np.allclose(np.diff(_idx[:, 0]), 1) and _idx[0, 0] == 0
    assert _empty.size == 0
    stacked_idx = pd.MultiIndex.from_product(
        (nirsraw_rhos, nirsraw_wavelengths, ("ac", "dc", "phase")), names=["rho", "wavelength", "variable"]
    )
    stacked_data = xr.DataArray(
        raw_data,
        dims=("time", "stacked_rho_wavelength_variable"),
        coords={
            "stacked_rho_wavelength_variable": stacked_idx,
        },
    )
    return (
        stacked_data.to_unstacked_dataset("stacked_rho_wavelength_variable", 2)
        .unstack("stacked_rho_wavelength_variable")
        .assign(
            aux=(("time", "rho"), aux),
            dark=(("time", "rho"), dark),
        )
    )
