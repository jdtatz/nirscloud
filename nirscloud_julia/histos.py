import typing

import numpy as np
import xarray as xr

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


def xr_vector_norm(x, dim, ord=None):
    return xr.apply_ufunc(
        np.linalg.norm, x, input_core_dims=[[dim]], kwargs={"ord": ord, "axis": -1}
    )


def plot_with_y_histogram(ax, x, y, smooth=True, hist_size="15%"):
    ax.plot(x, y, 'k')
    ax.margins(x=0)
    divider = make_axes_locatable(ax)
    ax_y_pdf = divider.append_axes("right", size=hist_size, pad=0, sharey=ax)
    if smooth:
        from scipy import stats
        y_lin = np.linspace(*ax.get_ylim(), 512)
        y_pdf = stats.gaussian_kde(y[np.isfinite(y)])
        ax_y_pdf.fill_betweenx(y_lin, 0, y_pdf(y_lin), facecolor="none", edgecolor=ax.spines["right"].get_edgecolor(), hatch="x", clip_on=False,)
    else:
        ax_y_pdf.hist(y, fc='none', ec='k', density=True, orientation='horizontal', bins=16)
    ax_y_pdf.set_xlim(0, None)
    ax_y_pdf.set_axis_off()
    ax_y_pdf.set_frame_on(False)
    return ax_y_pdf


def mpl_histo(ax, time, data, fiducial_vs, smooth=True, hist_size="15%",):
    ax_y_pdf = plot_with_y_histogram(ax, time, data, smooth=smooth, hist_size=hist_size)

    # ax.xaxis.set_major_locator(mpl.dates.AutoDateLocator())
    # ax.xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(mpl.dates.AutoDateLocator()))
    from datetime import timedelta
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda dt, pos: timedelta(seconds=dt)))

    for i, (v, lc) in enumerate(zip(fiducial_vs, ("r", "g", "b"))):
        w = 3
        ls = w * i, (w, 2 * w)
        ax.axhline(v, color=lc, ls=ls, zorder=0.5, label=v.fiducial.item())
        ax_y_pdf.axhline(v, color=lc, ls=ls, zorder=0.5, label=v.fiducial.item())
    return ax_y_pdf


def plot_histogramed_positioning(fig, fastrak_ds: xr.Dataset, measurements_ds=None, *, smooth=True, hist_size="15%", time_slice=slice(None), nirs_cmap: 'dict[str, typing.Any] | str | mpl.colors.Colormap'="Set2", nirs_alpha=0.4):
    if hasattr(fig, "add_gridspec"):
        # Is entire figure
        gs = fig.add_gridspec(4, 1)
    elif hasattr(fig, "subgridspec"):
        gs = fig.subgridspec(4, 1)
    else:
        raise TypeError(f"fig must be either a `Figure` or a `SubplotSpec`, not a {type(fig)}")
    axs = gs.subplots(sharex="col")
    y_pdf_axs = []

    time = fastrak_ds.time
    position = fastrak_ds.position

    time = time.sel(time=time_slice)
    position = position.sel(time=time_slice)

    t0 = time[0]
    time = (time - t0) / np.timedelta64(1, 's')

    location = fastrak_ds.coords["location"].item()
    axs[0].set_title(f"{location}-sensor positioning")
    # Position
    for ax, c in zip(axs[:-1], fastrak_ds.coords["cartesian_axes"]):
        v = position.sel(cartesian_axes=c)
        ax_y_pdf = mpl_histo(ax, time, v, fastrak_ds.coords["fiducial_position"].sel(fastrak_idx=1, cartesian_axes=c), smooth=smooth, hist_size=hist_size)
        y_pdf_axs.append(ax_y_pdf)
        ax.set_ylabel(f"position {c.item()} (cm)")

    # Rho
    ax = axs[-1]
    v = xr_vector_norm(position, dim="cartesian_axes")
    ax_y_pdf = mpl_histo(ax, time, v, xr_vector_norm(fastrak_ds.coords["fiducial_position"].sel(fastrak_idx=1), dim="cartesian_axes"), smooth=smooth, hist_size=hist_size)
    y_pdf_axs.append(ax_y_pdf)
    ax.set_xlabel("measurement timestamp")
    ax.set_ylabel(f"distance (cm)")

    # NIRS measurment highlights
    if measurements_ds is not None:
        measurement_locations = np.unique(measurements_ds.coords["measurement_location"])
        if isinstance(nirs_cmap, dict):
            colors = nirs_cmap
        else:
            cmap = mpl.cm.get_cmap(nirs_cmap, len(measurement_locations))
            colors = {l: cmap(i) for i, l in enumerate(measurement_locations)}
        start = (measurements_ds.nirs_start_time - t0) / np.timedelta64(1, 's')
        end = (measurements_ds.nirs_end_time - t0) / np.timedelta64(1, 's')
        for ax in axs:
            for span in zip(start, end):
                mloc = span[0].coords["measurement_location"].item()
                c = colors[mloc]
                ax.axvspan(*span, alpha=nirs_alpha, color=c, label=mloc)
    return gs, axs, y_pdf_axs


def histogramed_positioning_legend(fig):
    # axs[0].legend()
    # h, l = axs[0].get_legend_handles_labels()
    h, l = fig.axes[0].get_legend_handles_labels()
    # remove dups
    h, l = zip(*((handle, label) for i, (handle, label) in enumerate(zip(h, l)) if label not in l[:i]))
    lgd = fig.legend(h, l, bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    # lgd = fig.legend(h, l)
    return lgd
