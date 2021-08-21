import numpy as np
import xarray as xr

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


def xr_vector_norm(x, dim, ord=None):
    return xr.apply_ufunc(
        np.linalg.norm, x, input_core_dims=[[dim]], kwargs={"ord": ord, "axis": -1}
    )

def plot_with_hist(ax, x, y, smooth=True, hist_size="15%"):
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
        ax_y_pdf.hist(data, fc='none', ec='k', density=True, orientation='horizontal', bins=16)
    ax_y_pdf.set_xlim(0, None)
    ax_y_pdf.set_axis_off()
    ax_y_pdf.set_frame_on(False)
    return ax_y_pdf


def mpl_histo(ax, time, data, fiducial_vs, smooth=True, hist_size="15%",):
    ax_y_pdf = plot_with_hist(ax, time, data, smooth=smooth, hist_size=hist_size)

    # ax.xaxis.set_major_locator(mpl.dates.AutoDateLocator())
    # ax.xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(mpl.dates.AutoDateLocator()))
    from datetime import timedelta
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda dt, pos: timedelta(seconds=dt)))

    for i, (v, lc) in enumerate(zip(fiducial_vs, ("r", "g", "b"))):
        w = 3
        ls = w * i, (w, 2 * w)
        ax.axhline(v, color=lc, ls=ls, zorder=0.5, label=v.fiducial.item())
        ax_y_pdf.axhline(v, color=lc, ls=ls, zorder=0.5, label=v.fiducial.item())

def mpl_histos(fig, fastrak_ds: xr.Dataset, measurements_ds=None, *, smooth=True, hist_size="15%", time_slice=slice(None), nirs_cmap="Set2", nirs_alpha=0.4, location="head"):
    if hasattr(fig, "add_gridspec"):
        # Is entire figure
        gs = fig.add_gridspec(4, 1)
    elif hasattr(fig, "subgridspec"):
        gs = fig.subgridspec(4, 1)
    else:
        raise TypeError(f"fig must be either a `Figure` or a `SubplotSpec`, not a {type(fig)}")
    axs = gs.subplots(sharex="col")
    # axs = fig.subplots(4, 1, sharex="col")

    time = fastrak_ds.time
    position = fastrak_ds.position.sel(location=location)

    time = time.sel(time=time_slice)
    position = position.sel(time=time_slice)

    time = (time - time[0]) / np.timedelta64(1, 's')


    # Position
    for ax, c in zip(axs[:-1], fastrak_ds.coords["cartesian_axes"]):
        v = position.sel(cartesian_axes=c)
        mpl_histo(ax, time, v, fastrak_ds.fiducial_position.sel(fastrak_idx=1, cartesian_axes=c), smooth=smooth, hist_size=hist_size)
        ax.set_ylabel(f"position {c.item()} (cm)")

    # Rho
    ax = axs[-1]
    v = xr_vector_norm(position, dim="cartesian_axes")
    mpl_histo(ax, time, v, xr_vector_norm(fastrak_ds.fiducial_position.sel(fastrak_idx=1), dim="cartesian_axes"), smooth=smooth, hist_size=hist_size)
    ax.set_ylabel(f"distance (cm)")

    s = fastrak_ds.subject.item()
    d = fastrak_ds.date.dt.date.values # np.datetime_as_string(fastrak_ds.date.values, unit="D")
    axs[0].set_title(f"{location}-sensor positioning for {s} on {d}")

    if measurements_ds is not None:
        measurement_locations = np.unique(measurements_ds.coords["measurement_location"])
        cmap = mpl.cm.get_cmap(nirs_cmap, len(measurement_locations))
        colors = {l: cmap(i) for i, l in enumerate(measurement_locations)}
        start = (measurements_ds.nirs_start_time - fastrak_ds.time[0]) / np.timedelta64(1, 's')
        end = (measurements_ds.nirs_end_time - fastrak_ds.time[0]) / np.timedelta64(1, 's')
        for ax in axs:
            for span in zip(start, end):
                mloc = span[0].coords["measurement_location"].item()
                c = colors[mloc]
                ax.axvspan(*span, alpha=nirs_alpha, color=c, label=mloc)
    return gs, axs


def mpl_histos_legend(fig):
    # axs[0].legend()
    # h, l = axs[0].get_legend_handles_labels()
    h, l = fig.axes[0].get_legend_handles_labels()
    # remove dups
    h, l = zip(*((handle, label) for i, (handle, label) in enumerate(zip(h, l)) if label not in l[:i]))
    lgd = fig.legend(h, l, bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    # lgd = fig.legend(h, l)
    return lgd



def main():
    from io import StringIO
    # from IPython.display import SVG

    fig = mpl.figure.Figure(figsize=(24, 8), constrained_layout=True)
    t_max = fv.measurements.nirs_end_time.max().values + np.timedelta64(30, "s")
    gs = fig.add_gridspec(1, 2)
    _, _ = mpl_histos(gs[0, 0], fv.fastrak_ds, fv.measurements, time_slice=slice(None, t_max))
    _, _ = mpl_histos(gs[0, 1], fv.fastrak_ds, fv.measurements, time_slice=slice(None, t_max), location="nirs")
    lgd = mpl_histos_legend(fig)
    with StringIO() as f:
        fig.savefig(f, format="svg", bbox_extra_artists=(lgd,), bbox_inches='tight')
        # fig.savefig(f, format="svg")
        svg_str = f.getvalue()
    # return SVG(svg_str)
    return (svg_str)

