import datetime
import typing
from itertools import product

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d, axes3d
import numpy as np
import quaternion
import xarray as xr
from matplotlib.gridspec import GridSpecFromSubplotSpec, SubplotSpec
from scipy import interpolate
from scipy.spatial.transform import Rotation

from dual_quaternion import DualQuaternion
from nirscloud_julia.mpl_marker import marker_with_text


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


def auto_fig_size(height, width):
    return plt.figaspect(height / width)


def plot_measurment_trials(sb: SubplotSpec, measurement_ds: xr.Dataset, axes=('x', 'y', 'z'), ft_locs=("head", "nirs"), wavelength_indices=(0,), rho_idxs=(0, -1)):
    ntrial = len(measurement_ds.measurement)
    fastrak_time_s = measurement_ds.fastrak_time / np.timedelta64(1, 's')
    nirs_time_s = measurement_ds.time / np.timedelta64(1, 's')
    
    ft_idxs = tuple(product(axes, ft_locs))
    nirs_idxs = tuple(product(wavelength_indices, rho_idxs))

    gs = GridSpecFromSubplotSpec(len(ft_idxs) + len(nirs_idxs), ntrial, sb)
    axs = gs.subplots(sharex="col")

    for ax, m in zip(axs[0], measurement_ds.measurement):
        trial = m.item().rsplit('_')[-1]
        ax.set_title(f"trial {trial}")
    for i, axs_ in enumerate(axs.T):
        # b/c there are duplicated measurment_ids, somehow???
        ds_ = measurement_ds.isel(measurement=i)
        for ax, (d, l) in zip(axs_, ft_idxs):
            v = ds_.position.sel(cartesian_axes=d, location=l)
            ax.plot(fastrak_time_s, v.values)
            # ax.set_xlabel("fastrak time (s)")
            if i == 0:
                ax.set_ylabel(f"{l}-{d} ({v.units})")
        axs_ = axs_[len(ft_idxs):]
        for ax, (w_idx, r_idx) in zip(axs_, nirs_idxs):
            ax.plot(nirs_time_s, ds_.ac.isel(wavelength=w_idx, rho=r_idx))
            if i == 0:
                w = ds_.wavelength[w_idx]
                r = ds_.rho[r_idx]
                ax.set_ylabel(f"ac [λ={w.item()}, ρ={r.item()}]")
        axs_[-1].set_xlabel("time (s)")
    return gs, axs


def interactive_bokeh_measurement_time_plot(measurement_ds, ft_locs=("head", "nirs"), wavelength_indices=(-1,), rho_indices=(0, -1)):
    from bokeh.layouts import gridplot
    from bokeh.models import Panel, Tabs
    from bokeh.plotting import figure

    fastrak_time_s = np.asarray(measurement_ds.fastrak_time / np.timedelta64(1, 's'))
    nirs_time_s = np.asarray(measurement_ds.time / np.timedelta64(1, 's'))

    wavelengths = [measurement_ds.wavelength[i] for i in wavelength_indices]
    rhos = [measurement_ds.rho[i] for i in rho_indices]

    cartesian_axes = 'x', 'y', 'z'
    ft_idxs = tuple(product(cartesian_axes, ft_locs))
    nirs_idxs = tuple(product(wavelengths, rhos))

    # b/c there are duplicated measurment_ids, somehow???
    measurement_trials = [(k, [trials.isel(measurement=i) for i in range(len(trials.measurement_trial))]) for k, trials in measurement_ds.groupby("measurement_location")]

    m_g_children = [(k, [
        [{"x": fastrak_time_s, "y": trial_ds.position.sel(cartesian_axes=d, location=l), "y_axis_label": f"{l}-{d} ({trial_ds.position.units})", "title": f"Trial {trial_ds.measurement_trial.item()}"} for d, l in ft_idxs] + [{"x": nirs_time_s, "y": trial_ds.ac.sel(wavelength=w, rho=r), "y_axis_label": f"ac [λ={w.item()}, ρ={r.item()}]", "title": f"Trial {trial_ds.measurement_trial.item()}"} for w, r in nirs_idxs]
        for trial_ds in trials
    ]) for k, trials in measurement_trials]


    for _, g_children in m_g_children:
        for i, col in enumerate(g_children):
            for child in col[1:]:
                del child["title"]
            for child in col[:-1]:
                child["x_axis_location"] = None
            if i > 0:
                for child in col:
                    del child["y_axis_label"]

    def line_plot(d):
        x = d.pop("x")
        y = np.asarray(d.pop("y").values)
        p = figure(**d)
        p.line(x, y)
        # Adjust to have min 1mm height
        p.y_range.bounds = "auto"
        return p

    m_g_children = [(k, [[line_plot(d) for d in col] for col in g_children]) for k, g_children in m_g_children]

    # fix
    for _, g_children in m_g_children:
        for col in g_children:
            col[-1].x_range.bounds = "auto"
            col[-1].xaxis.axis_label = "time (s)"
            for child in col[:-1]:
                child.x_range = col[-1].x_range
    #             child.xgrid.visible = False
    #             child.ygrid.visible = False
    #         col[-1].xgrid.visible = False
    #         col[-1].ygrid.visible = False

    m_children = [(k, list(map(list, zip(*g_children)))) for k, g_children in m_g_children]
    
    grids = [(k, gridplot(children=children, plot_width=800, plot_height=200,)) for k, children in m_children]

    tabs = [Panel(child=v, title=k) for k, v in grids]
    return Tabs(tabs=tabs)


def minimal_rotation(a: np.ndarray, b: np.ndarray) -> Rotation:
    """
    Create minimal rotation R, such that R @ a = b and R @ c = c where c = a x b

    Antiparallel a and b is undefined, but parallel a and b is Identity
    """
    assert np.shape(a)[-1] == 3
    assert np.shape(b)[-1] == 3
    assert np.allclose(np.linalg.norm(a, axis=-1), 1)
    assert np.allclose(np.linalg.norm(b, axis=-1), 1)
    v = np.cross(a, b, axis=-1)
    c = np.inner(a, b)
    rot = Rotation.from_quat(np.concatenate((v, 1 + c[..., None]), axis=-1))
    assert np.allclose(rot.apply(a), b)
    assert np.allclose(rot.apply(v), v)
    return rot


class OrientationVisualization(art3d.Poly3DCollection):
    @staticmethod
    def tri_pts() -> np.ndarray:
        zb = 0
        v0 = -1,  0, zb
        v1 =  0, -1, zb
        v2 =  1,  0, zb
        v3 =  0,  1, zb
        v4 =  0,  0,  1
        return np.array((v0, v1, v2, v3, v4))

    _tri_indices = np.array((
        # base
        (0, 1, 3),
        (2, 3, 1),
        # sides
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (0, 3, 4),
    ))

    @property
    def polyhedron_faces(self) -> np.ndarray:
        if not hasattr(self, "_polyhedron_faces"):
            faces = OrientationVisualization.tri_pts()[OrientationVisualization._tri_indices]
            z = np.array((0, 0, 1))
            rot_to_initial = minimal_rotation(z, self.initial_direction)
            self._polyhedron_faces = np.apply_along_axis(rot_to_initial.apply, -1, faces)
        return self._polyhedron_faces
    
    def orientated_faces(self, orientation: Rotation) -> np.ndarray:
        assert orientation.single
        return np.apply_along_axis(orientation.apply, -1, self.polyhedron_faces)

    def __init__(self, orientation: Rotation = Rotation.identity(), initial_direction: np.ndarray=np.array((0, 0, 1)), **kwargs):
        self.initial_direction = initial_direction
        verts = self.orientated_faces(orientation)
        super().__init__(verts, zsort="max", **kwargs)

    def set_orientation(self, orientation: Rotation):
        verts = self.orientated_faces(orientation)
        super().set_verts(verts, closed=True)
        if self.axes.M is not None:
            super().do_3d_projection()
        return


def create_rotating_polyhedron(ax: axes3d.Axes3D, initial_direction: np.ndarray=np.array((0, 0, 1))):
    colors = 'black', 'black', '#e66101','#fdb863', '#5e3c99', '#b2abd2'
    # colors = 'black', 'black', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'

    # ax.axis('off')
    lim = 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    polyhedron = OrientationVisualization(initial_direction=initial_direction, linewidths=1, alpha=1, animated=False)
    polyhedron.set_facecolor(colors)
    polyhedron.set_edgecolor('k')
    ax.add_collection3d(polyhedron)

    def draw(orientation: Rotation):
        nonlocal polyhedron
        polyhedron.set_orientation(orientation)
        return polyhedron,
    return draw



def visualize_path(ax: axes3d.Axes3D, path: np.ndarray):
    segments = np.concatenate([path[:-1, None], path[1:, None]], axis=1)
    lc = art3d.Line3DCollection(segments, linewidths=3, capstyle='round', joinstyle='round')
    lc.set_array(np.linspace(0, 1, len(path)))
    ax.add_collection3d(lc)
    ax.auto_scale_xyz(*path.T)
    # lc = ax.scatter(*positions.T, c=np.arange(len(positions)), edgecolors='none', alpha=1)
    return lc



class FastrakVisualization:
    def __init__(self, fastrak_ds: xr.Dataset, nirs_ds: xr.Dataset):
        self.fastrak_ds   = fastrak_ds
        self.nirs_ds      = nirs_ds
        self.measurements = combine_measurements_ds(self.fastrak_ds, self.nirs_ds)

    def interactive_3d_mean_plot(self, fig, cmap = 'Set2'):
        # fastrak_mean_measurements = self.measurements[["measurement_location", "position", "orientation"]].mean(dim="fastrak_time") # Idk why this doesn't work
        fastrak_mean_measurements = self.measurements[["measurement_location", "position", "orientation"]].map(lambda v: v.mean(dim="fastrak_time") if "fastrak_time" in v.dims else v)
        fastrak_mean_measurements_grouped = dict(fastrak_mean_measurements.groupby("measurement_location"))
        cmap = mpl.cm.get_cmap(cmap, len(fastrak_mean_measurements_grouped))

        # Absolute
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        s = 100
        for i, (loc, trials) in enumerate(fastrak_mean_measurements_grouped.items()):
            c = cmap(i)
            for j in range(len(trials.measurement)):
                trial = trials.isel(measurement=j)
                s_marker = marker_with_text("s", trial.coords["measurement_trial"].item())
                o_marker = marker_with_text("o", trial.coords["measurement_trial"].item())
                ax.scatter(*trial.position.sel(location='head').T, color=c, depthshade=False, s=s, edgecolor='k', marker=s_marker)
                ax.scatter(*trial.position.sel(location='nirs').T, color=c, depthshade=False, s=s, edgecolor='k', marker=o_marker)

        for l in ("nose", "left_ear", "right_ear"):
            marker = marker_with_text("o", l[0].upper())
            p = self.measurments.coords["fiducial_position"].sel(fiducial=l, fastrak_idx=1)
            ax.scatter(*p, color="white", depthshade=False, s=s, edgecolor='k', marker=marker)

        # Relative
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(0, 0, 0, s=s, color='k')
        for i, (loc, trials) in enumerate(fastrak_mean_measurements_grouped.items()):
            # ax.scatter(*trials.position.sel(location='relative').T, color=cmap(i), depthshade=False, s=s, edgecolor='k')
            c = cmap(i)
            for j in range(len(trials.measurement)):
                trial = trials.isel(measurement=j)
                marker = marker_with_text("o", trial.coords["measurement_trial"].item())
                ax.scatter(*trial.position.sel(location='relative').T, color=c, depthshade=False, s=s, edgecolor='k', marker=marker)

        for l in ("nose", "left_ear", "right_ear"):
            marker = marker_with_text("o", l[0].upper())
            h_p = self.measurments.coords["fiducial_position"].sel(fiducial=l, fastrak_idx=1)
            h_o = self.measurments.coords["fiducial_orientation"].sel(fiducial=l, fastrak_idx=1)
            h_dq = DualQuaternion.from_rigid_position(h_p, quaternion.as_quat_array(h_o))
            p = self.measurments.coords["fiducial_position"].sel(fiducial=l, fastrak_idx=0)
            p = h_dq.apply(p)
            ax.scatter(*p, color="white", depthshade=False, s=s, edgecolor='k', marker=marker)

        # Colobar
        n = len(fastrak_mean_measurements_grouped)
        ticks = np.arange(1 / (2 * n), 1, step = 1 / n)
        clb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ticks=ticks)
        clb.set_ticklabels(list(fastrak_mean_measurements_grouped.keys()))
        return ax, clb

    def positioning_with_histogram_plot(self, fig, *, smooth=True, hist_size="15%", tz: 'typing.Optional[str | datetime.tzinfo]'=None, nirs_cmap: 'dict[str, typing.Any] | str | mpl.colors.Colormap'="Set2", nirs_alpha=0.4, use_nirs_time_subset_for_lim=False):
        from .histos import (histogramed_positioning_legend,
                             plot_histogramed_positioning)
        t_max = self.measurements.nirs_end_time.max().values + np.timedelta64(30, "s")
        gs = fig.add_gridspec(1, 2)
        histo_kwargs = dict(time_slice=slice(None, t_max), smooth=smooth, hist_size=hist_size, nirs_cmap=nirs_cmap, nirs_alpha=nirs_alpha, tz=tz)
        _ = plot_histogramed_positioning(gs[0, 0], self.fastrak_ds.sel(location="head"), self.measurements, **histo_kwargs)
        _ = plot_histogramed_positioning(gs[0, 1], self.fastrak_ds.sel(location="nirs"), self.measurements, **histo_kwargs, use_nirs_time_subset_for_lim=use_nirs_time_subset_for_lim)
        lgd = histogramed_positioning_legend(fig)
        fig.suptitle(f"Subject {self.fastrak_ds.subject.item()} on {np.datetime_as_string(self.fastrak_ds.time[0], unit='D', timezone=tz, casting='unsafe')}")
        return lgd

    def interactive_measurement_time_plot(self, ft_locs=("head", "nirs", "relative"), wavelength_indices=(0,), rho_idxs=(0, -1), figsize=(16, 8)):
        msrmnt_loc_dict = dict(self.measurements.groupby("measurement_location"))

        def interactive_update_wrapper(key):
            measurement_ds = msrmnt_loc_dict[key]
            fig.clf()
            gs = fig.add_gridspec(1, 1)
            sub_gs, sub_axs = plot_measurment_trials(gs[0, 0], measurement_ds, ft_locs=ft_locs, wavelength_indices=wavelength_indices, rho_idxs=rho_idxs)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        choice = widgets.Dropdown(options=msrmnt_loc_dict.keys(), description="Measurement Location", value=None)

        # out = widgets.interactive_output(interactive_update_wrapper, {'measurement_ds': choice})

        plt.ioff()
        fig = plt.figure(constrained_layout=True, figsize=figsize,)
        plt.ion()
        fig.canvas.header_visible = False
        # fig.canvas.layout.min_height = '400px'

        choice.observe(lambda change: interactive_update_wrapper(change['new']), 'label')
        choice.value = next(iter(msrmnt_loc_dict.keys()))
        gui = widgets.VBox([choice, fig.canvas])
        return gui

    def interactive_3d_time_plot(self, measurement: str, *, figsize=auto_fig_size(1, 2)):
        example = self.nirs_ds.nirs_start_time.sel(measurement=measurement)
        t_slice = slice(example, example + example.duration)
        test = self.fastrak_ds.sel(time=t_slice, location='relative')
        test_rots = Rotation.from_quat(quaternion.as_float_array(test.orientation)[..., [1, 2, 3, 0]])

        plt.ioff()
        fig = plt.figure(figsize=figsize)
        plt.ion()

        fig.suptitle(str(example.measurement.values))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        visualize_path(ax, test.position.values)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        # visualize_path(ax, test_rots.apply(np.array((0, 0, 1))))
        draw = create_rotating_polyhedron(ax)
        draw(test_rots[0])

        sel_times = ((test.time - example) / np.timedelta64(1, 's')).values
        sel = widgets.SelectionSlider(options=[(f"{int(t*100)/100} s", t) for t in sel_times])

        def update(index):
            draw(test_rots[index])
        sel.observe(lambda change: update(change['new']), 'index')
        sel.index = 0

        gui = widgets.VBox([sel, fig.canvas])
        return gui


def get_bokeh_theme():
    from importlib.resources import path

    from bokeh.themes import Theme
    
    with path(__package__, "nirs_bokeh_theme.yml") as fpath:
        return Theme(filename=fpath)


def apply_bokeh_theme(doc = None):
    if doc is None:
        import bokeh.io
        doc = bokeh.io.curdoc()
    doc.theme = get_bokeh_theme()
