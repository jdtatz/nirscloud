from typing import Optional
import numpy as np
from numpy.polynomial import Polynomial
import scipy
import scipy.stats as stats
import scipy.optimize as optimize
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import arviz as az
import ipywidgets as widgets
from functools import lru_cache

import os

try:
    os.environ["AESARA_FLAGS"] = "device=cuda,floatX=float32"
    import pymc as pm

    _has_pymc = True
    _has_pymc_gwr = False
except ImportError:
    try:
        os.environ["THEANO_FLAGS"] = "device=cuda,floatX=float32"
        import pymc3 as pm

        _has_pymc = _has_pymc_gwr = True
    except ImportError:
        _has_pymc = _has_pymc_gwr = False


try:
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfb = tfp.bijectors
    tfd = tfp.distributions
    tfk = tfp.math.psd_kernels
    has_tfp = True
except ImportError:
    has_tfp = False


def xr_polynomial_fit(x: xr.DataArray, y: xr.DataArray, *fit_dims: str, degree: int = 1):
    def fitter(x, y):
        poly, ([resid], rank, sv, rcond) = Polynomial.fit(x.flat, y.flat, deg=degree, full=True)
        r2 = 1 - resid / (y.size * y.var())
        return poly, poly.convert().coef, resid, r2, rank, sv, rcond

    poly, coef, resid, r2, rank, sv, rcond = xr.apply_ufunc(
        fitter,
        x,
        y,
        input_core_dims=[fit_dims, fit_dims],
        output_core_dims=[(), ("degree",), (), (), (), ("degree",), ()],
        vectorize=True,
    )
    return xr.Dataset(
        {
            "poly": poly,
            "coef": coef,
            "resid": resid,
            "r2": r2,
            "rank": rank,
            "sv": sv,
            "rcond": rcond,
        },
        coords={"degree": (["degree"], coef.degree.values)},
    )


def phase_stats(phase_da: xr.DataArray, dim="time") -> xr.Dataset:
    cphase = np.exp(1j * phase_da.astype(np.float64))
    z_avg = cphase.mean(dim=dim)
    z_conj_avg = np.conjugate(cphase).mean(dim=dim)
    R_avg_2 = z_avg * z_conj_avg
    N = cphase.sizes[dim]
    R_e_2 = N / (N - 1) * (R_avg_2 - 1 / N)
    mean = np.angle(z_avg)
    circ_var = 1 - np.sqrt(R_e_2)
    std_dev = np.sqrt(np.log(1 / R_e_2))
    cphase_ds = xr.Dataset(
        {
            "mean": (circ_var.dims, mean, {"standard_name": "circular_mean", "long_name": "$\\overline{z}$"}),
            "variance": circ_var.real,
            "std_dev": std_dev.real,
        }
    )
    cphase_ds.variance.attrs["standard_name"] = "circular_variance"
    cphase_ds.variance.attrs["long_name"] = "$Var(z)$"
    cphase_ds.std_dev.attrs["standard_name"] = "circular_standard_deviation"
    cphase_ds.std_dev.attrs["long_name"] = "$S(z)$"
    return cphase_ds


def non_outlier_mask(da: xr.DataArray, dim="time", threshold: float = 2.25):
    mad = np.abs(da - da.median(dim="time"))
    # k = 1 / stats.norm.ppf(3/4)
    k = 1.482602218505602
    mad_level = k * mad.median(dim="time")
    wo_outliers = (mad / mad_level) <= threshold
    return wo_outliers


def rolling_trend(da: xr.DataArray, dim="time", window_size=20, min_periods: "Optional[int]" = 1):
    return da.rolling({dim: window_size}, center=True, min_periods=min_periods).mean()


def rolling_detrend_mean_var(
    da: xr.DataArray,
    dim="time",
    window_size=20,
    min_periods: "Optional[int]" = 1,
    outlier_threshold: "Optional[float]" = 2.25,
):
    trend = rolling_trend(da, dim=dim, window_size=window_size, min_periods=min_periods)
    detrended = da - (trend - da.mean(dim=dim))
    if outlier_threshold is not None:
        detrended = detrended.where(non_outlier_mask(detrended, threshold=outlier_threshold))
    return detrended.mean(dim=dim), detrended.var(dim=dim)


if _has_pymc_gwr:

    @lru_cache()
    def pymc_smooth_random_walk_model(n_time_pts: int):
        LARGE_NUMBER = 1e5

        model = pm.Model()
        with model:
            smoothing_param = pm.Data("s", 0.9)
            mu = pm.Normal("mu", sigma=LARGE_NUMBER)
            tau = pm.Exponential("tau", 1.0 / LARGE_NUMBER)
            z_tau = tau / (1.0 - smoothing_param)
            z = pm.GaussianRandomWalk("z", mu=mu, tau=z_tau, shape=n_time_pts)
            true = pm.Data("true", np.zeros(n_time_pts))
            obs = pm.Normal("obs", mu=z, tau=tau / smoothing_param, observed=true)
        return model

    def pymc_smooth_random_walk_trend(da: xr.DataArray, dim="time", smoothing_param=0.9, progressbar=False):
        time = da.coords[dim]
        (n_time_pts,) = time.shape
        model = pymc_smooth_random_walk_model(n_time_pts)

        with model:
            pm.set_data(
                dict(
                    s=smoothing_param,
                    true=da.values,
                )
            )
            res = pm.find_MAP(vars=[model["z"]], method="L-BFGS-B", progressbar=False)
            return res["z"]


if _has_pymc:

    @lru_cache()
    def pymc_gaussian_process_model(n_time_pts: int):
        model = pm.Model()
        with model:
            naive_mean = pm.Data("naive_mean", 0)
            index_pts = pm.Data("index_pts", np.zeros((n_time_pts, 1)))
            true = pm.Data("true", np.zeros(n_time_pts))

            G0 = pm.Flat("G0") + naive_mean
            # G0 = pm.Normal("G0", mu=v.values.mean(), sigma=3 * v.values.std())
            # G0 = v.values[:3].mean()
            gp_mean = pm.gp.mean.Constant(G0)
            # gp_mean = pm.gp.mean.Zero()

            # informative lengthscale prior
            ℓ = pm.Gamma("ℓ", alpha=2, beta=1)
            # weakly informative over the covariance function scale, and noise scale
            η = pm.HalfCauchy("η", beta=1)
            gp_cov = η ** 2 * pm.gp.cov.Matern52(1, ℓ)
            # gp_cov = pm.gp.cov.Exponential(1, ls=1e-4)

            gp = pm.gp.Marginal(mean_func=gp_mean, cov_func=gp_cov)
            sigma = pm.HalfCauchy("sigma", beta=3)
            z = gp.marginal_likelihood("z", X=index_pts, y=true, noise=sigma)
        return model

    def pymc_gaussian_process_trend_trace(
        da: xr.DataArray, dim="time", progressbar=False, sample_kwargs: dict = None
    ) -> az.InferenceData:
        time = da.coords[dim]
        if time.dtype.type is np.timedelta64:
            time = time / np.timedelta64(1, "s")
        (n_time_pts,) = time.shape
        model = pymc_gaussian_process_model(n_time_pts)
        sample_kwargs = sample_kwargs or {}

        with model:
            pm.set_data(
                dict(
                    naive_mean=da.values.mean(),
                    index_pts=time.values[:, None],
                    true=da.values,
                )
            )
            return pm.sample(**sample_kwargs, return_inferencedata=True)


if has_tfp:
    tfp_2_az_mapping = {
        "target_log_prob": "lp",
        "leapfrogs_taken": "tree_size",
        "has_divergence": "diverging",
        "energy": "energy",
        "log_accept_ratio": "mean_tree_accept",
    }

    #     @lru_cache()
    def tfp_gaussian_process_model(
        da: xr.DataArray, dim="time", float_dtype=np.float32
    ) -> tfp.experimental.distributions.JointDistributionPinned:
        time = da.coords[dim]
        if time.dtype.type is np.timedelta64:
            time = time / np.timedelta64(1, "s")

        observation_index_points_ = time.values[:, None].astype(float_dtype)
        observations_ = da.values.astype(float_dtype)

        stationary_mean_mu = observations_.mean()
        stationary_mean_sigma = observations_.std()  # * float_dtype(4)

        @tfd.JointDistributionCoroutineAutoBatched
        def model():
            unshifted_stationary_mean = yield tfd.Normal(
                loc=float_dtype(0), scale=float_dtype(1), name="unshifted_stationary_mean"
            )
            stationary_mean = stationary_mean_mu + stationary_mean_sigma * unshifted_stationary_mean
            amplitude = yield tfd.LogNormal(loc=float_dtype(0), scale=float_dtype(1), name="amplitude")
            length_scale = yield tfd.LogNormal(loc=float_dtype(0), scale=float_dtype(1), name="length_scale")
            kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
            observation_noise_variance = yield tfd.LogNormal(
                loc=float_dtype(0), scale=float_dtype(1), name="observation_noise_variance"
            )
            observations = yield tfd.GaussianProcess(
                mean_fn=lambda _: stationary_mean,
                kernel=kernel,
                index_points=observation_index_points_,
                observation_noise_variance=observation_noise_variance,
                name="observations",
            )

        return model.experimental_pin(observations=observations_)

    def tfp_gaussian_process_trace(
        da: xr.DataArray,
        dim="time",
        float_dtype=np.float32,
        n_chains: int = 8,
        step_size: float = 0.1,
        n_results: int = 1_000,
        n_burnin_steps: int = 1_000,
        proportion_adaption_steps: float = 0.8,
        n_prior_samples: Optional[int] = None,
    ) -> az.InferenceData:
        # TODO replace later
        from tensorflow_probability.python.internal import unnest

        inference_data_dict = {}

        target_model = tfp_gaussian_process_model(da, dim=dim, float_dtype=float_dtype)

        if n_prior_samples is not None and n_prior_samples > 0:
            model_unpinned_keys = tuple(target_model.event_shape._asdict().keys())
            model_pinned_keys = tuple(target_model.pins.keys())
            prior_sample = target_model.distribution.sample(n_prior_samples)
            prior_samples = {k: getattr(prior_sample, k) for k in model_unpinned_keys}
            prior_predictive = {k: getattr(prior_sample, k) for k in model_pinned_keys}
            inference_data_dict.update(
                observed_data={"observations": da.values},
                prior={k: v[tf.newaxis, ...] for k, v in prior_samples.items()},
                prior_predictive={k: v[tf.newaxis, ...] for k, v in prior_predictive.items()},
                coords={dim: da.coords[dim].values},
                dims={"observations": list(da.dims)},
            )

        bij = target_model.experimental_default_event_space_bijector()
        # pulled_back_shape = bij.inverse_event_shape(target_model.event_shape)

        nuts = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target_model.log_prob, step_size=tf.cast(step_size, tf.float32)
        )

        transformed_nuts = tfp.mcmc.TransformedTransitionKernel(nuts, bijector=bij)

        transformed_adaptive_nuts = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=transformed_nuts,
            num_adaptation_steps=int(proportion_adaption_steps * n_burnin_steps),
            #     target_accept_prob=tf.cast(0.75, tf.float32)
        )

        # uniform_init = tf.nest.map_structure(
        #     lambda s: tf.random.uniform(tf.concat([[n_chains], s], axis=0), -2., 2.),
        #     pulled_back_shape
        # )
        # initial_state = bij.forward(uniform_init)

        random_sample, _ = target_model.sample_and_log_weight(n_chains)
        initial_state = random_sample

        # Speed up sampling by tracing with `tf.function`.
        @tf.function(autograph=True, jit_compile=True)
        def do_sampling():
            return tfp.mcmc.sample_chain(
                kernel=transformed_adaptive_nuts,
                current_state=initial_state,
                num_results=n_results,
                num_burnin_steps=n_burnin_steps,
                trace_fn=lambda current_state, kernel_results: kernel_results,
            )

        mcmc_samples, mcmc_diagnostic = do_sampling()

        # TFP -> arviz conversion code from https://jeffpollock9.github.io/bayesian-workflow-with-tfp-and-arviz/
        # arviz expects shapes to be of the form (chain, draw, *shape), but
        # TFP gives (draw, chain, *shape)
        sample_stats = {
            az_k: np.swapaxes(unnest.get_innermost(mcmc_diagnostic, tfp_k), 1, 0)
            for tfp_k, az_k in tfp_2_az_mapping.items()
        }
        posterior = {k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples._asdict().items()}
        inference_data_dict.update(
            sample_stats=sample_stats,
            posterior=posterior,
        )
        return az.from_dict(**inference_data_dict)


# FP012 on 12/8/21 , MetaID("RvS54nMYT8G8lYE1nwLz0A")
# B099 on 3/12/21 , MetaID("kv8EQwYWQ96vCW6KCfKM6Q")
def calibrate(
    calib_measurement_ds: xr.Dataset, detrend_mean_var_fn=rolling_detrend_mean_var, **detrend_fn_kwargs
) -> xr.Dataset:
    calib_ds = calib_measurement_ds
    cphase_ds = phase_stats(calib_ds.phase)

    dc_no_dark = calib_ds.dc - calib_ds.dark.mean(dim="time")
    dc_detrend_mean, dc_detrend_var = detrend_mean_var_fn(dc_no_dark, **detrend_fn_kwargs)

    # AC already has no dark
    ac_no_dark = calib_ds.ac
    ac_detrend_mean, ac_detrend_var = detrend_mean_var_fn(ac_no_dark, **detrend_fn_kwargs)

    calib_stats_ds = xr.Dataset(
        dict(
            dc_mean=dc_detrend_mean.assign_attrs(long_name="Detrended DC mean"),
            dc_var=dc_detrend_var.assign_attrs(long_name="Detrended DC variance"),
            ac_mean=ac_detrend_mean.assign_attrs(long_name="Detrended AC mean"),
            ac_var=ac_detrend_var.assign_attrs(long_name="Detrended AC variance"),
            phase_mean=cphase_ds["mean"],
            phase_var=cphase_ds["variance"],
        )
    )

    def calibrate_and_apply_acdc(ds):
        # dc_cal_ds = xr_polynomial_fit(ds.dc_mean, ds.dc_var, "measurement", "wavelength", degree=1)
        dc_cal_ds = xr_polynomial_fit(ds.dc_mean, ds.dc_var, *ds.dims, degree=1)
        dc_cal = dc_cal_ds.coef.sel(degree=1).drop_vars("degree")
        ac_cal_ds = xr_polynomial_fit(ds.ac_mean, ds.ac_var, *ds.dims, degree=1)
        ac_cal = ac_cal_ds.coef.sel(degree=1).drop_vars("degree")

        phase_cal_ds = xr_polynomial_fit(np.log(ds.ac_mean / ac_cal.values), np.log(ds.phase_var), *ds.dims, degree=1)
        phase_cal = phase_cal_ds.coef.sel(degree=1).drop_vars("degree")

        return ds.assign(
            dc_cal=dc_cal,
            dc_cal_poly=dc_cal_ds.poly,
            dc_cal_r2=dc_cal_ds.r2,
            ac_cal=ac_cal,
            ac_cal_poly=ac_cal_ds.poly,
            ac_cal_r2=ac_cal_ds.r2,
            phase_cal=phase_cal,
            phase_cal_poly=phase_cal_ds.poly,
            phase_cal_r2=phase_cal_ds.r2,
            dc_cal_mean=(ds.dc_mean / dc_cal).assign_attrs(long_name="Calibrated DC mean"),
            dc_cal_var=(ds.dc_var / dc_cal ** 2).assign_attrs(long_name="Calibrated DC variance"),
            ac_cal_mean=(ds.ac_mean / ac_cal).assign_attrs(long_name="Calibrated AC mean"),
            ac_cal_var=(ds.ac_var / ac_cal ** 2).assign_attrs(long_name="Calibrated AC variance"),
        )
    
    def scalar_safe_groupby_map(ds, group, f):
        return f(ds) if ds[group].shape == () else ds.groupby(group).map(f)

    return calib_stats_ds.groupby("rho").map(lambda ds: scalar_safe_groupby_map(ds, "gain", calibrate_and_apply_acdc))
