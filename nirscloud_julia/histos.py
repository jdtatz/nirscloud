import datetime
import typing

import numpy as np
import xarray as xr

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

__all__ = [
    "plot_with_y_histogram",
    "mpl_histo",
    "plot_histogramed_positioning",
    "histogramed_positioning_legend",
    "mokeypatch_matplotlib_constrained_layout",
]


def xr_vector_norm(x, dim, ord=None):
    return xr.apply_ufunc(np.linalg.norm, x, input_core_dims=[[dim]], kwargs={"ord": ord, "axis": -1})


def plot_with_y_histogram(ax, x, y, smooth=True, hist_size="15%"):
    ax.plot(x, y, "k")
    ax.margins(x=0)
    divider = make_axes_locatable(ax)
    ax_y_pdf = divider.append_axes("right", size=hist_size, pad=0, sharey=ax)
    if smooth:
        from scipy import stats

        y_lin = np.linspace(*ax.get_ylim(), 512)
        y_pdf = stats.gaussian_kde(y[np.isfinite(y)])
        ax_y_pdf.fill_betweenx(
            y_lin,
            0,
            y_pdf(y_lin),
            facecolor="none",
            edgecolor=ax.spines["right"].get_edgecolor(),
            hatch="x",
        )
    else:
        ax_y_pdf.hist(y, fc="none", ec="k", density=True, orientation="horizontal", bins=16)
    ax_y_pdf.set_xlim(0, None)
    ax_y_pdf.set_axis_off()
    ax_y_pdf.set_frame_on(False)
    return ax_y_pdf


def mpl_histo(
    ax,
    time,
    data,
    constants: typing.Dict[str, typing.Tuple[float, str]],
    smooth=True,
    hist_size="15%",
    tz: "typing.Optional[str | datetime.tzinfo]" = None,
):
    ax_y_pdf = plot_with_y_histogram(ax, time, data, smooth=smooth, hist_size=hist_size)

    loc = mpl.dates.AutoDateLocator(tz=tz)
    ax.xaxis.set_major_locator(loc)
    fmt = mpl.dates.ConciseDateFormatter(loc, tz=tz)
    ax.xaxis.set_major_formatter(fmt)

    n = len(constants)
    for i, (label, (v, color)) in enumerate(constants.items()):
        w = n
        ls = w * i, (w, (n - 1) * w)
        ax.axhline(v, color=color, ls=ls, zorder=0.5, label=label)
        ax_y_pdf.axhline(v, color=color, ls=ls, zorder=0.5, label=label)
    return ax_y_pdf


def plot_histogramed_positioning(
    fig_or_spec: "Figure | SubplotSpec",
    fastrak_ds: xr.Dataset,
    measurements_ds=None,
    *,
    smooth=True,
    hist_size="15%",
    time_slice: slice = slice(None),
    fiducial_idx: int = 1,
    tz: "typing.Optional[str | datetime.tzinfo]" = None,
    nirs_cmap: "dict[str, typing.Any] | str | mpl.colors.Colormap" = "Set2",
    nirs_alpha=0.4,
    use_nirs_time_subset_for_lim=False,
):
    if isinstance(fig_or_spec, Figure):
        # Is entire figure
        gs = fig_or_spec.add_gridspec(4, 1)
    elif isinstance(fig_or_spec, SubplotSpec):
        gs = fig_or_spec.subgridspec(4, 1)
    else:
        raise TypeError(f"fig must be either a `Figure` or a `SubplotSpec`, not a {type(fig_or_spec)}")
    axs = gs.subplots(sharex="col")
    fig = gs.figure
    y_pdf_axs = []

    time = fastrak_ds.time
    position = fastrak_ds.position

    time = time.sel(time=time_slice)
    position = position.sel(time=time_slice)

    location = fastrak_ds.coords["location"].item()
    axs[0].set_title(f"{location}-sensor positioning")
    axs[-1].set_xlabel("measurement timestamp")

    values, fiducial_values, labels = zip(
        *(
            *(
                (
                    position.sel(cartesian_axes=c),
                    fastrak_ds.coords["fiducial_position"].sel(cartesian_axes=c),
                    f"position {c.item()} (cm)",
                )
                for c in fastrak_ds.coords["cartesian_axes"]
            ),
            (
                xr_vector_norm(position, dim="cartesian_axes"),
                xr_vector_norm(
                    fastrak_ds.coords["fiducial_position"],
                    dim="cartesian_axes",
                ),
                "distance (cm)",
            ),
        )
    )
    fiducial_constants = ({v.fiducial.item(): (v, c) for v, c in zip(fv, ("r", "g", "b"))} for fv in fiducial_values)

    for ax, v, fv, l in zip(axs, values, fiducial_constants, labels):
        ax_y_pdf = mpl_histo(ax, time, v, fv, smooth=smooth, hist_size=hist_size, tz=tz)
        y_pdf_axs.append(ax_y_pdf)
        ax.set_ylabel(l)

    # Fix-up subplots
    fig.align_ylabels(axs)

    # NIRS measurment highlights
    if measurements_ds is not None:
        measurement_locations = np.unique(measurements_ds.coords["measurement_location"])
        if isinstance(nirs_cmap, dict):
            colors = nirs_cmap
        else:
            cmap = mpl.cm.get_cmap(nirs_cmap, len(measurement_locations))
            colors = {l: cmap(i) for i, l in enumerate(measurement_locations)}
        time_slices = [
            slice(s, e)
            for s, e in zip(
                measurements_ds.nirs_start_time.values,
                measurements_ds.nirs_end_time.values,
            )
        ]
        for ax in axs:
            for (tspan, mloc) in zip(time_slices, measurements_ds.coords["measurement_location"].values):
                c = colors[mloc]
                ax.axvspan(tspan.start, tspan.stop, alpha=nirs_alpha, color=c, label=mloc)
        if use_nirs_time_subset_for_lim:
            for ax, v, fv in zip(axs, values, fiducial_values):
                minv = min(min(v.sel(time=ts).min() for ts in time_slices), fv.min())
                maxv = max(max(v.sel(time=ts).max() for ts in time_slices), fv.max())
                margin = (maxv - minv) * ax.margins()[1]
                ax.set_ylim(minv - margin, maxv + margin)
    return gs, axs, y_pdf_axs


_mokeypatched_matplotlib_constrained_layout = False


def histogramed_positioning_legend(fig: Figure):
    # axs[0].legend()
    # h, l = axs[0].get_legend_handles_labels()
    h, l = fig.axes[0].get_legend_handles_labels()
    # remove dups
    h, l = zip(*((handle, label) for i, (handle, label) in enumerate(zip(h, l)) if label not in l[:i]))
    lgd = fig.legend(h, l, loc="upper right", borderaxespad=0)
    if _mokeypatched_matplotlib_constrained_layout:
        lgd._outside = True
    else:
        lgd.set_bbox_to_anchor((1, 1))
    # lgd = fig.legend(h, l)
    return lgd


def mokeypatch_matplotlib_constrained_layout():
    """Monkeypatch `ENH: allow fig.legend outside axes... #19743` to fix Figure.legend in constrained layouts
    https://github.com/matplotlib/matplotlib/pull/19743"""
    from functools import wraps
    import matplotlib._constrained_layout as constrained_layout

    global _mokeypatched_matplotlib_constrained_layout
    if _mokeypatched_matplotlib_constrained_layout:
        return

    make_layout_margins = constrained_layout._make_layout_margins

    @wraps(make_layout_margins)
    def wrapped_make_layout_margins(fig, renderer, *args, w_pad=0, h_pad=0, hspace=0, wspace=0, **kwargs):
        ret = make_layout_margins(
            fig,
            renderer,
            *args,
            w_pad=w_pad,
            h_pad=h_pad,
            hspace=hspace,
            wspace=wspace,
            **kwargs,
        )
        # make margins for figure-level legends:
        for leg in fig.legends:
            inv_trans_fig = None
            if getattr(leg, "_outside", None) and leg._bbox_to_anchor is None:
                if inv_trans_fig is None:
                    inv_trans_fig = fig.transFigure.inverted().transform_bbox
                bbox = inv_trans_fig(leg.get_tightbbox(renderer))
                w = bbox.width + 2 * w_pad
                h = bbox.height + 2 * h_pad
                if (leg._loc in (3, 4) and leg._outside == "lower") or (leg._loc == 8):
                    fig._layoutgrid.edit_margin_min("bottom", h)
                elif (leg._loc in (1, 2) and leg._outside == "upper") or (leg._loc == 9):
                    fig._layoutgrid.edit_margin_min("top", h)
                elif leg._loc in (1, 4, 5, 7):
                    fig._layoutgrid.edit_margin_min("right", w)
                elif leg._loc in (2, 3, 6):
                    fig._layoutgrid.edit_margin_min("left", w)
        return ret

    constrained_layout._make_layout_margins = wrapped_make_layout_margins
    _mokeypatched_matplotlib_constrained_layout = True
