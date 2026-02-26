from rtm_pymmcore.agents.abstract_agent import Agent
from abc import abstractmethod
import pandas as pd
import dataclasses
from typing import List, Optional
import gpax
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm
import numpy as np
import gpax.utils
from gpax.models.gp import ExactGP

from gpax import acquisition
import gpax.utils
import matplotlib.pyplot as plt


@dataclasses.dataclass
class BO_Parameter:
    name: str
    bounds: tuple
    param_type: str = "float"  # 'int' or 'float'
    spacing: float = 10.0
    log_scale: bool = False
    initial_value: float = None


@dataclasses.dataclass
class BO_Covariate:
    name: str
    bounds: tuple = None
    log_scale: bool = False


@dataclasses.dataclass
class BO_Objective:
    name: str
    goal: str  # 'minimize' or 'maximize'
    log_scale: bool = False


class StandardScalerBounds:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.log_scale = None

    def fit(self, X: np.ndarray, bounds: list = None, log_scale: list = None):
        """Fit scaler. If `feature_names` and `bounds` are provided, use bounds for those features.

        For a feature with bounds (min, max) we set mean = (min + max) / 2 and
        std = (max - min) / 2 so that values at the bounds map to roughly +/-1.
        """
        X = jnp.asarray(X)
        self.log_scale = log_scale

        if log_scale is not None:
            for i, use_log in enumerate(log_scale):
                if use_log:
                    X = X.at[:, i].set(jnp.log(X[:, i]))

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        if bounds is not None and log_scale is not None:
            for i, bound in enumerate(bounds):
                if bound is not None and log_scale[i]:
                    minv, maxv = jnp.log(bound[0]), jnp.log(bound[1])
                    self.mean_ = self.mean_.at[i].set((minv + maxv) / 2.0)
                    self.std_ = self.std_.at[i].set((maxv - minv) / 2.0)
                elif bound is not None:
                    minv, maxv = bound
                    self.mean_ = self.mean_.at[i].set((minv + maxv) / 2.0)
                    self.std_ = self.std_.at[i].set((maxv - minv) / 2.0)

        self.std_ = jnp.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X):
        X = jnp.asarray(X)
        # Selektive log-Transformation pro Feature
        if self.log_scale is not None:
            for i, use_log in enumerate(self.log_scale):
                if use_log:
                    X = X.at[:, i].set(jnp.log(X[:, i]))
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, bounds=None, log_scale=None):
        return self.fit(X, bounds=bounds, log_scale=log_scale).transform(X)

    def inverse_transform(self, X):
        """Reverse the scaling and log transformation.

        Reverses the operations in transform(): first unnormalizes, then reverses log.
        """
        X = jnp.asarray(X)
        # Reverse normalization: X_scaled * std + mean
        X = X * self.std_ + self.mean_
        # Reverse selective log-Transformation pro Feature
        if self.log_scale is not None:
            for i, use_log in enumerate(self.log_scale):
                if use_log:
                    X = X.at[:, i].set(jnp.exp(X[:, i]))
        return X


class BOptGPAX(Agent):
    def __init__(
        self,
        microscope,
        parameters_to_optimize: List[BO_Parameter],
        objective_metric: BO_Objective,
        bo_covariates: List[BO_Covariate] = [],
        n_iterations: int = 10,
        true_marginalization: bool = False,
        acquisition_function: str = "ucb",  # 'ucb' | 'ei'
        ucb_beta: float = 4.0,
        ei_xi: float = 0.0,
        ei_num_samples: int = 16,
        ei_noiseless: bool = True,
        n_cov_samples: int = 10,
        penalty: Optional[str] = None,
        penalty_factor: float = 1.0,
        prior=None,
        verbose: bool = False,
    ):

        self.n_iterations = n_iterations
        self.iteration = 0
        self.microscope = microscope
        self.parameters_to_optimize = parameters_to_optimize
        self.objective_metric = objective_metric
        self.model = None  # Placeholder for the GP model

        self.fovs = None

        self.x_covariate = None
        self._seed = 42

        self.x = None
        self.y = None

        self.x_total_linespace = self._generate_total_parameter_space()
        self.x_unmeasured = self._generate_total_parameter_space()
        self.x_performed_experiments = None  # single array of all performed experiments
        self.bo_covariates = bo_covariates
        self.true_marginalization = true_marginalization

        self.acquisition_function = acquisition_function.lower().strip()
        if self.acquisition_function not in {"ucb", "ei"}:
            raise ValueError(
                f"Unsupported acquisition_function='{acquisition_function}'. Use 'ucb' or 'ei'."
            )
        self.ucb_beta = float(ucb_beta)
        self.ei_xi = float(ei_xi)
        self.ei_num_samples = int(ei_num_samples)
        self.ei_noiseless = bool(ei_noiseless)
        self.n_cov_samples = int(n_cov_samples)

        # Penalty settings for discouraging re-evaluation at recent points
        # Options: None, 'delta', 'inverse_distance'
        if penalty is not None:
            penalty = penalty.lower().strip()
            if penalty not in {"delta", "inverse_distance"}:
                raise ValueError(
                    f"Unsupported penalty='{penalty}'. Use None, 'delta', or 'inverse_distance'."
                )
        self.penalty = penalty
        self.penalty_factor = float(penalty_factor)

        # Tracks which acquisition was actually used in the most recent BO round
        # (can differ from self.acquisition_function if we fall back).
        self._acquisition_used_this_round = self.acquisition_function

        self.verbose = verbose
        self._iteration_means = []

    def _apply_penalty(
        self,
        acq: jnp.ndarray,
        x_scaled: jnp.ndarray,
        recent_points: np.ndarray,
        penalty: str,
        penalty_factor: float,
    ) -> jnp.ndarray:
        """Apply penalty to acquisition function based on recent points.

        Both ``x_scaled`` and ``recent_points`` must be in the **same**
        coordinate space (e.g. both normalised with the same scaler) so
        that Euclidean distances are meaningful.

        Args:
            acq: Acquisition function values
            x_scaled: Input points where acquisition was computed (scaled)
            recent_points: Recently evaluated points (scaled, same space)
            penalty: Type of penalty ('delta' or 'inverse_distance')
            penalty_factor: Strength of penalty for 'inverse_distance'

        Returns:
            Penalized acquisition values
        """
        recent_scaled = jnp.asarray(recent_points)

        if penalty == "delta":
            # Infinite penalty for exactly matching points
            # For each recent point, set acquisition to -inf for matches
            for rp in recent_scaled:
                # Check if any point matches (within numerical tolerance)
                distances = jnp.linalg.norm(x_scaled - rp, axis=1)
                mask = distances < 1e-8  # Very small tolerance for "exact" match
                acq = jnp.where(mask, -jnp.inf, acq)

        elif penalty == "inverse_distance":
            # Penalty based on distance to nearest recent point
            # For each candidate point, find minimum distance to any recent point
            min_distances = jnp.full(x_scaled.shape[0], jnp.inf)

            for rp in recent_scaled:
                distances = jnp.linalg.norm(x_scaled - rp, axis=1)
                min_distances = jnp.minimum(min_distances, distances)

            # Penalty term: higher penalty for closer points
            # Add small epsilon to avoid division by zero
            epsilon = 1e-6
            penalty_term = 1.0 / (min_distances + epsilon)

            # Apply penalty: acq_penalized = acq / (1 + penalty_factor * penalty_term)
            acq = acq / (1.0 + penalty_factor * penalty_term)

        return acq

    def _is_flat_acquisition(self, acq_values, range_tol: float = 1e-6) -> bool:
        acq = jnp.asarray(acq_values)
        finite = jnp.isfinite(acq)
        if not bool(jnp.any(finite)):
            return True
        acq_f = acq[finite]
        acq_max = jnp.max(acq_f)
        acq_min = jnp.min(acq_f)
        return bool((acq_max - acq_min) < range_tol)

    def _generate_total_parameter_space(self):
        param_grids = []
        for param in self.parameters_to_optimize:
            if param.spacing is None:
                spacing = 1.0
            else:
                spacing = param.spacing
            if param.param_type == "int":
                grid = np.arange(param.bounds[0], param.bounds[1] + spacing, spacing)
            elif param.param_type == "float":
                grid = np.arange(param.bounds[0], param.bounds[1] + spacing, spacing)
            else:
                raise ValueError(f"Unsupported parameter type: {param.param_type}")
            param_grids.append(grid)

        mesh = np.meshgrid(*param_grids, indexing="ij")
        flat_mesh = [m.flatten() for m in mesh]
        total_param_space = np.column_stack(flat_mesh)
        return total_param_space

    def add_fovs(self, fovs: list):
        self.fovs = fovs

    def _extract_x_from_df(self, df: pd.DataFrame) -> np.ndarray:
        X = []
        for param in self.parameters_to_optimize:
            X.append(df[param.name].values)
        for covariate in self.bo_covariates:
            X.append(df[covariate.name].values)
        return np.column_stack(X)

    def _extract_y_from_df(self, df: pd.DataFrame) -> np.ndarray:
        y = df[self.objective_metric.name].values
        # if self.objective_metric.goal == 'maximize':
        #     y = -y  # Convert maximization to minimization
        return y.reshape(-1, 1)

    def _get_bounds_and_log_scale(self):
        bounds = []
        log_scale = []
        for param in self.parameters_to_optimize:
            bounds.append(param.bounds)
            log_scale.append(param.log_scale)
        for covariate in self.bo_covariates:
            bounds.append(covariate.bounds)
            log_scale.append(covariate.log_scale)
        return bounds, log_scale

    def _extract_spread_points(self, X, k=3):
        rng = np.random.default_rng(self._seed)
        n = X.shape[0]
        # Startpunkt zufällig
        idx = [rng.integers(n)]
        # Distanz zum Startpunkt
        d = np.linalg.norm(X - X[idx[0]], axis=1)

        for _ in range(1, k):
            # wähle Punkt mit größtem Abstand zum bisherigen Set
            next_idx = np.argmax(d)
            idx.append(next_idx)
            # update: Mindestabstand zu ausgewählten Punkten
            d = np.minimum(d, np.linalg.norm(X - X[next_idx], axis=1))

        idx_array = np.array(idx)
        return np.array(X[idx_array]), idx

    def _select_initial_samples(self, k=3):
        x_scaler = StandardScalerBounds()
        bounds, log_scale = self._get_bounds_and_log_scale()

        X_scaled = x_scaler.fit_transform(
            self.x_unmeasured, bounds=bounds, log_scale=log_scale
        )
        X_selected_scaled, selected_indices = self._extract_spread_points(X_scaled, k=k)
        X_selected = x_scaler.inverse_transform(X_selected_scaled)
        self.x_unmeasured = np.delete(self.x_unmeasured, selected_indices, axis=0)
        self.x_performed_experiments = (
            np.concatenate([self.x_performed_experiments, X_selected], axis=0)
            if self.x_performed_experiments is not None
            else X_selected
        )

        selected_parameter_dicts = []
        for selected_row in np.asarray(X_selected):
            selected_parameter_dicts.append(
                {
                    param.name: selected_row[i]
                    for i, param in enumerate(self.parameters_to_optimize)
                }
            )

        return selected_parameter_dicts, selected_indices

    def _compute_mc_ei(
        self,
        rng_key,
        gp_model,
        x_scaled,
        best_f_scaled,
        maximize: bool,
        batch_size=1000,
        xi: float = 0.0,
        penalty: Optional[str] = None,
        recent_points: Optional[np.ndarray] = None,
        penalty_factor: float = 1.0,
    ):
        """Expected Improvement aligned with GPax EI formula.

        GPax uses mean/variance of the predictive distribution:
        EI = sigma * (phi(u) + u * Phi(u)), with u = (mean - best_f) / sigma.
        We estimate mean/var from posterior samples to match GPax's HMC handling.

        Args:
            penalty: Penalty type ('delta' or 'inverse_distance') to discourage re-evaluation
                     at or near recently evaluated points.
            recent_points: Array of recently evaluated points (scaled coordinates).
            penalty_factor: Factor controlling penalty strength for 'inverse_distance'.
        """
        mean_pred, y_sampled = gp_model.predict_in_batches(
            rng_key,
            x_scaled,
            batch_size=batch_size,
            n=self.ei_num_samples,
            noiseless=self.ei_noiseless,
        )

        mean_broadcast = mean_pred[None, None, :]
        y_sampled = jnp.where(jnp.isfinite(y_sampled), y_sampled, mean_broadcast)

        mean = jnp.mean(y_sampled, axis=(0, 1))
        var = jnp.var(y_sampled, axis=(0, 1))

        # Numerical floor for stability (GPax does not clamp, but we avoid NaNs)
        sigma = jnp.sqrt(jnp.maximum(var, 1e-12))

        u = (mean - (best_f_scaled + xi)) / sigma
        if not maximize:
            u = -u

        ei = sigma * (jnorm.pdf(u) + u * jnorm.cdf(u))
        ei = jnp.where(jnp.isfinite(ei), ei, 0.0)

        # Apply penalty if specified
        if penalty is not None and recent_points is not None and len(recent_points) > 0:
            ei = self._apply_penalty(
                ei, x_scaled, recent_points, penalty, penalty_factor
            )

        return ei

    def _compute_mc_ucb(
        self,
        rng_key,
        gp_model,
        x_scaled,
        maximize: bool,
        beta: float = 4.0,
        batch_size=1000,
        penalty: Optional[str] = None,
        recent_points: Optional[np.ndarray] = None,
        penalty_factor: float = 1.0,
    ):
        """Monte-Carlo UCB aligned with GPax formula using posterior mean/variance.

        Args:
            penalty: Penalty type ('delta' or 'inverse_distance') to discourage re-evaluation
                     at or near recently evaluated points.
            recent_points: Array of recently evaluated points (scaled coordinates).
            penalty_factor: Factor controlling penalty strength for 'inverse_distance'.
        """
        mean_pred, y_sampled = gp_model.predict_in_batches(
            rng_key,
            x_scaled,
            batch_size=batch_size,
            n=self.ei_num_samples,
            noiseless=self.ei_noiseless,
        )

        mean_broadcast = mean_pred[None, None, :]
        y_sampled = jnp.where(jnp.isfinite(y_sampled), y_sampled, mean_broadcast)

        mean = jnp.mean(y_sampled, axis=(0, 1))
        var = jnp.var(y_sampled, axis=(0, 1))
        delta = jnp.sqrt(beta * var)
        ucb = mean + delta if maximize else -(mean - delta)
        ucb = jnp.where(jnp.isfinite(ucb), ucb, 0.0)

        # Apply penalty if specified
        if penalty is not None and recent_points is not None and len(recent_points) > 0:
            ucb = self._apply_penalty(
                ucb, x_scaled, recent_points, penalty, penalty_factor
            )

        return ucb

    def _compute_robust_acq(
        self, rng_key, gp_model, x_grid, c_samples, x_scaler, y_scaled, batch_size=1000
    ):
        """Compute robust acquisition function by marginalizing over covariate samples.

        For **EI** a *covariate-conditional incumbent* is used: the
        improvement threshold is the best GP-predicted mean over the
        controllable-parameter grid *at each covariate sample separately*.
        This prevents a single favourable-covariate observation from
        suppressing EI at less favourable covariate values.

        For **UCB** no incumbent is needed, so the standard formula is
        applied per (x, c) pair and then averaged.

        Penalty (if enabled) is applied **after** marginalisation, using
        only the controllable-parameter dimensions (properly scaled).

        Args:
            rng_key: JAX random key
            gp_model: Trained GP model
            x_grid: Grid of controllable parameters (N_grid, n_params) - unscaled
            c_samples: Samples from covariate distribution (N_mc, n_covariates) - unscaled
            x_scaler: StandardScalerBounds instance for input scaling
            y_scaled: Scaled training outputs (unused when EI uses model-based incumbent)
            batch_size: Number of points to evaluate at once

        Returns:
            acq_values: Acquisition values marginalized over covariates (N_grid,)
        """
        n_grid = x_grid.shape[0]
        n_mc = c_samples.shape[0] if c_samples.ndim > 1 else len(c_samples)

        # Reshape c_samples to (N_mc, n_covariates) if needed
        if c_samples.ndim == 1:
            c_samples = c_samples.reshape(-1, 1)

        # Build full (x, c) evaluation batch ----------------------------------
        x_repeated = jnp.repeat(x_grid, n_mc, axis=0)  # (N_grid*N_mc, n_params)
        c_tiled = jnp.tile(c_samples, (n_grid, 1))  # (N_grid*N_mc, n_covs)
        x_full_batch = jnp.hstack(
            [x_repeated, c_tiled]
        )  # (N_grid*N_mc, n_params+n_covs)
        x_full_batch_scaled = x_scaler.transform(x_full_batch)

        n_total = x_full_batch_scaled.shape[0]
        # Clamp batch_size so gpax split_in_batches doesn't fail on small inputs
        batch_size = min(batch_size, n_total)
        print(
            f"Computing robust acquisition over {n_total} scenarios "
            f"({n_grid} grid points x {n_mc} covariate samples)..."
        )

        maximize = self.objective_metric.goal == "maximize"

        # GP predictions for the full batch ------------------------------------
        mean_pred, y_sampled = gp_model.predict_in_batches(
            rng_key,
            x_full_batch_scaled,
            batch_size=batch_size,
            n=self.ei_num_samples,
            noiseless=self.ei_noiseless,
        )

        mean_broadcast = mean_pred[None, None, :]
        y_sampled = jnp.where(jnp.isfinite(y_sampled), y_sampled, mean_broadcast)

        mean_flat = jnp.mean(y_sampled, axis=(0, 1))  # (N_grid*N_mc,)
        var_flat = jnp.var(y_sampled, axis=(0, 1))  # (N_grid*N_mc,)

        # Reshape to (N_grid, N_mc) for per-covariate operations
        mean_matrix = mean_flat.reshape(n_grid, n_mc)
        var_matrix = var_flat.reshape(n_grid, n_mc)

        # Compute acquisition per (grid point, covariate sample) ---------------
        if self.acquisition_function == "ei":
            # Covariate-conditional incumbent: for each covariate sample the
            # best predicted mean over the grid serves as the improvement
            # threshold.  This is fairer than a single global best_f because
            # a high-covariate observation does not inflate the bar for
            # low-covariate scenarios.
            if maximize:
                best_f_per_cov = jnp.max(mean_matrix, axis=0)  # (N_mc,)
            else:
                best_f_per_cov = jnp.min(mean_matrix, axis=0)

            print(
                f"  best_f per covariate (scaled): "
                f"min={float(jnp.min(best_f_per_cov)):.6f}, "
                f"max={float(jnp.max(best_f_per_cov)):.6f}"
            )

            sigma_matrix = jnp.sqrt(jnp.maximum(var_matrix, 1e-12))

            # u shape: (N_grid, N_mc), best_f_per_cov broadcasts along grid dim
            u = (mean_matrix - (best_f_per_cov[None, :] + self.ei_xi)) / sigma_matrix
            if not maximize:
                u = -u

            acq_matrix = sigma_matrix * (jnorm.pdf(u) + u * jnorm.cdf(u))
            acq_matrix = jnp.where(jnp.isfinite(acq_matrix), acq_matrix, 0.0)
        else:
            # UCB – no incumbent needed
            delta = jnp.sqrt(self.ucb_beta * var_matrix)
            if maximize:
                acq_matrix = mean_matrix + delta
            else:
                acq_matrix = -(mean_matrix - delta)
            acq_matrix = jnp.where(jnp.isfinite(acq_matrix), acq_matrix, 0.0)

        # Marginalise: average over covariate samples --------------------------
        robust_acq = jnp.mean(acq_matrix, axis=1)  # (N_grid,)

        # Penalty (controllable params only, properly scaled) ------------------
        if self.penalty is not None and self.x_performed_experiments is not None:
            n_ctrl = len(self.parameters_to_optimize)
            recent_ctrl_raw = self.x_performed_experiments[:, :n_ctrl]

            # Scale controllable params using the first n_ctrl dims of the scaler
            ctrl_log = (
                x_scaler.log_scale[:n_ctrl] if x_scaler.log_scale is not None else None
            )
            ctrl_mean = x_scaler.mean_[:n_ctrl]
            ctrl_std = x_scaler.std_[:n_ctrl]

            def _scale_ctrl(X):
                X = jnp.asarray(X)
                if ctrl_log is not None:
                    for i, use_log in enumerate(ctrl_log):
                        if use_log:
                            X = X.at[:, i].set(jnp.log(X[:, i]))
                return (X - ctrl_mean) / ctrl_std

            x_grid_ctrl_scaled = _scale_ctrl(x_grid)
            recent_ctrl_scaled = _scale_ctrl(jnp.asarray(recent_ctrl_raw))

            robust_acq = self._apply_penalty(
                robust_acq,
                x_grid_ctrl_scaled,
                np.asarray(recent_ctrl_scaled),
                self.penalty,
                self.penalty_factor,
            )

        print(
            f"  Robust acq stats: min={float(jnp.min(robust_acq)):.6f}, "
            f"max={float(jnp.max(robust_acq)):.6f}"
        )

        return robust_acq

    def _determine_next_parameters(
        self, df_results: pd.DataFrame, verbose=False
    ) -> dict:

        # IMPORTANT: df_results already contains the full history.
        # Do NOT concatenate df_results into self.x/self.y on every iteration,
        # otherwise data gets duplicated (15 -> 35 -> 60 ...), which can cause
        # numerical issues (NaNs) in GP training and acquisition.

        x_scaler = StandardScalerBounds()
        bounds, log_scale = self._get_bounds_and_log_scale()

        y_scaler = StandardScalerBounds()
        y_log_scale = [self.objective_metric.log_scale]

        x_raw = self._extract_x_from_df(df_results)
        y_raw = self._extract_y_from_df(df_results)

        # Print the exact x/y data used for this BO round (raw values)
        # if not df_results.empty:
        #     df_train = df_results[x_cols + [y_col]].copy().reset_index(drop=True)

        #     # Helpful aggregated view: multiple samples per (x1,x2)
        #     group_cols = [p.name for p in self.parameters_to_optimize]
        #     agg_dict = {
        #         y_col: ["count", "mean", "std", "min", "max"],
        #     }
        #     for cov in self.bo_covariates:
        #         agg_dict[cov.name] = ["mean", "std", "min", "max"]
        #     df_summary = df_results.groupby(group_cols, dropna=False).agg(agg_dict)
        #     df_summary.columns = ["_".join(col).strip() for col in df_summary.columns.to_flat_index()]
        #     df_summary = df_summary.reset_index()
        #     print(f"Training dataframe summary (grouped by {group_cols}) for BO round {self.iteration + 1}:")
        #     print(df_summary)
        # else:
        #     # print(f"Training dataframe is empty for BO round {self.iteration + 1}.")

        x = x_scaler.fit_transform(x_raw, bounds=bounds, log_scale=log_scale)
        y = y_scaler.fit_transform(y_raw, bounds=None, log_scale=y_log_scale)

        # # Check for NaN/Inf in training data
        # print("Training data check:")
        # print(f"  x shape: {x.shape}, y shape: {y.shape}")
        # print(f"  x has NaN: {jnp.any(jnp.isnan(x))}, has Inf: {jnp.any(jnp.isinf(x))}")
        # print(f"  y has NaN: {jnp.any(jnp.isnan(y))}, has Inf: {jnp.any(jnp.isinf(y))}")
        # print(f"  y stats: min={float(jnp.min(y)):.6f}, max={float(jnp.max(y)):.6f}, mean={float(jnp.mean(y)):.6f}, std={float(jnp.std(y)):.6f}")

        rng_key, rng_key_predict = gpax.utils.get_keys()
        gp_model = gpax.ExactGP(kernel="Matern", input_dim=x.shape[1])

        gp_model.fit(
            rng_key, X=x, y=y, progress_bar=True, num_warmup=200, num_samples=1000
        )

        # Store model and scalers for post-hoc analysis / 3D visualization
        self.model = gp_model
        self._x_scaler = x_scaler
        self._y_scaler = y_scaler
        self._rng_key_predict = rng_key_predict

        # Only consider points not yet measured
        x_grid_ctrl = self.x_unmeasured.copy()

        acquisition_used = self.acquisition_function

        if len(self.bo_covariates) > 0:
            # Robust acquisition: sample covariates from observed distribution
            cov_samples_list = []
            for covariate in self.bo_covariates:
                cov_vals = df_results[covariate.name].values
                if cov_vals.size == 0:
                    cov_samples = np.array([0.0])
                else:
                    # Sample from observed covariate distribution (with replacement)
                    n_cov_samples = self.n_cov_samples
                    rng = np.random.default_rng(self._seed)
                    cov_samples = rng.choice(cov_vals, size=n_cov_samples, replace=True)
                cov_samples_list.append(cov_samples)

            # Create mesh of covariate samples
            if len(cov_samples_list) == 1:
                cov_grid = cov_samples_list[0].reshape(-1, 1)
            else:
                cov_mesh = np.meshgrid(*cov_samples_list, indexing="ij")
                cov_grid = np.stack([m.ravel() for m in cov_mesh], axis=1)

            # Compute robust acquisition marginalized over covariates
            acq_values_total = self._compute_robust_acq(
                rng_key_predict, gp_model, x_grid_ctrl, cov_grid, x_scaler, y
            )

            # Fallback: EI can become identically zero if the model is overconfident.
            if acquisition_used == "ei" and self._is_flat_acquisition(
                acq_values_total, range_tol=1e-6
            ):
                print(
                    "  EI is flat (max-min < 1e-6). Falling back to UCB for this round."
                )
                acquisition_used = "ucb"
                old_acq = self.acquisition_function
                try:
                    self.acquisition_function = "ucb"
                    acq_values_total = self._compute_robust_acq(
                        rng_key_predict, gp_model, x_grid_ctrl, cov_grid, x_scaler, y
                    )
                finally:
                    self.acquisition_function = old_acq
            x_total_with_cov = x_grid_ctrl
        else:
            # No covariates: acquisition on controllable grid directly
            x_total_scaled = x_scaler.transform(x_grid_ctrl)
            maximize = self.objective_metric.goal == "maximize"

            # Get recent points for penalty (scaled to match x_total_scaled)
            recent_ctrl = None
            if self.penalty is not None and self.x_performed_experiments is not None:
                n_ctrl_params = len(self.parameters_to_optimize)
                recent_ctrl_raw = self.x_performed_experiments[:, :n_ctrl_params]
                recent_ctrl = np.asarray(
                    x_scaler.transform(jnp.asarray(recent_ctrl_raw))
                )

            if self.acquisition_function == "ei":
                best_f_scaled = jnp.max(y) if maximize else jnp.min(y)
                acq_values_total = self._compute_mc_ei(
                    rng_key_predict,
                    gp_model,
                    x_total_scaled,
                    best_f_scaled=best_f_scaled,
                    maximize=maximize,
                    batch_size=1000,
                    xi=self.ei_xi,
                    penalty=self.penalty,
                    recent_points=recent_ctrl,
                    penalty_factor=self.penalty_factor,
                )

                if self._is_flat_acquisition(acq_values_total, range_tol=1e-6):
                    print(
                        "  EI is flat (max-min < 1e-6). Falling back to UCB for this round."
                    )
                    acquisition_used = "ucb"
                    acq_values_total = self._compute_mc_ucb(
                        rng_key_predict,
                        gp_model,
                        x_total_scaled,
                        maximize=maximize,
                        beta=self.ucb_beta,
                        batch_size=1000,
                        penalty=self.penalty,
                        recent_points=recent_ctrl,
                        penalty_factor=self.penalty_factor,
                    )
            else:
                acq_values_total = self._compute_mc_ucb(
                    rng_key_predict,
                    gp_model,
                    x_total_scaled,
                    maximize=maximize,
                    beta=self.ucb_beta,
                    batch_size=1000,
                    penalty=self.penalty,
                    recent_points=recent_ctrl,
                    penalty_factor=self.penalty_factor,
                )
            x_total_with_cov = x_grid_ctrl

            self._acquisition_used_this_round = acquisition_used

        next_measurement_idx = jnp.argmax(acq_values_total)
        next_parameters = np.asarray(x_grid_ctrl[int(next_measurement_idx)])

        # Remove chosen point from unmeasured set to avoid repeating it
        self.x_unmeasured = np.delete(
            self.x_unmeasured, int(next_measurement_idx), axis=0
        )
        self.x_performed_experiments = (
            np.concatenate(
                [self.x_performed_experiments, next_parameters.reshape(1, -1)], axis=0
            )
            if self.x_performed_experiments is not None
            else next_parameters.reshape(1, -1)
        )

        next_parameters_dict = {}
        for i, param in enumerate(self.parameters_to_optimize):
            next_parameters_dict[param.name] = next_parameters[i]

        if verbose:
            # For plotting we need acquisition values on the *full* controllable grid,
            # because _plot_mock_example reshapes into (len(unique_x1), len(unique_x2)).
            x_plot_ctrl = self.x_total_linespace.copy()

            if len(self.bo_covariates) > 0:
                # Reuse acquisition values from the optimization call;
                # only compute the (few) already-measured grid points.
                diffs = np.abs(x_plot_ctrl[:, None, :] - x_grid_ctrl[None, :, :])
                missing_mask = ~np.any(np.all(diffs < 1e-10, axis=2), axis=1)
                x_missing = x_plot_ctrl[missing_mask]

                if len(x_missing) > 0:
                    if acquisition_used == "ucb":
                        old_acq = self.acquisition_function
                        try:
                            self.acquisition_function = "ucb"
                            acq_missing = self._compute_robust_acq(
                                rng_key_predict,
                                gp_model,
                                x_missing,
                                cov_grid,
                                x_scaler,
                                y,
                            )
                        finally:
                            self.acquisition_function = old_acq
                    else:
                        acq_missing = self._compute_robust_acq(
                            rng_key_predict,
                            gp_model,
                            x_missing,
                            cov_grid,
                            x_scaler,
                            y,
                        )
                    acq_values_plot = np.empty(len(x_plot_ctrl))
                    acq_values_plot[~missing_mask] = np.asarray(acq_values_total)
                    acq_values_plot[missing_mask] = np.asarray(acq_missing)
                    acq_values_plot = jnp.array(acq_values_plot)
                else:
                    acq_values_plot = acq_values_total
                x_plot_with_cov = x_plot_ctrl
            else:
                # Compute recent points for penalty (only needed without covariates;
                # with covariates the penalty is applied inside _compute_robust_acq)
                recent_ctrl_plot = None
                if (
                    self.penalty is not None
                    and self.x_performed_experiments is not None
                ):
                    n_ctrl_params = len(self.parameters_to_optimize)
                    recent_ctrl_raw = self.x_performed_experiments[:, :n_ctrl_params]
                    recent_ctrl_plot = np.asarray(
                        x_scaler.transform(jnp.asarray(recent_ctrl_raw))
                    )

                x_plot_scaled = x_scaler.transform(x_plot_ctrl)
                maximize = self.objective_metric.goal == "maximize"
                if acquisition_used == "ei":
                    best_f_scaled = jnp.max(y) if maximize else jnp.min(y)
                    acq_values_plot = self._compute_mc_ei(
                        rng_key_predict,
                        gp_model,
                        x_plot_scaled,
                        best_f_scaled=best_f_scaled,
                        maximize=maximize,
                        batch_size=1000,
                        xi=self.ei_xi,
                        penalty=self.penalty,
                        recent_points=recent_ctrl_plot,
                        penalty_factor=self.penalty_factor,
                    )
                else:
                    acq_values_plot = self._compute_mc_ucb(
                        rng_key_predict,
                        gp_model,
                        x_plot_scaled,
                        maximize=maximize,
                        beta=self.ucb_beta,
                        batch_size=1000,
                        penalty=self.penalty,
                        recent_points=recent_ctrl_plot,
                        penalty_factor=self.penalty_factor,
                    )
                x_plot_with_cov = x_plot_ctrl

            self._plot_mock_example(
                df_results,
                x_scaler,
                y_scaler,
                gp_model,
                rng_key_predict,
                x_plot_with_cov,
                acq_values_plot,
                next_parameters,
                x_unmeasured_at_computation=x_grid_ctrl,
            )

        return next_parameters_dict

    def _get_ground_truth_grid(self, unique_x1, unique_x2, cov_mean):
        """Compute ground truth on a (x1, x2) grid.

        Returns an array of shape (len(unique_x1), len(unique_x2)) or None
        if no analytical ground truth is available (in which case the plot
        will show a convergence subplot instead).
        """
        return compute_ground_truth(unique_x1, unique_x2, cov_mean)

    def _plot_mock_example(
        self,
        df_results: pd.DataFrame,
        x_scaler,
        y_scaler,
        gp_model,
        rng_key_predict,
        x_total_with_cov,
        acq_values_total,
        next_point: Optional[np.ndarray] = None,
        x_unmeasured_at_computation: Optional[np.ndarray] = None,
    ):
        """Plot measured data, model predictions, and acquisition function.

        Creates 4 subplots:
        1. Measured points (x1, x2) with y as color, covariate as text annotations
        2. Ground truth (if available) or GP model uncertainty
        3. Acquisition function marginalized over covariate
        4. GP predicted landscape marginalized over covariate
        """

        # Extract controllable parameters grid
        x_total_ctrl = self.x_total_linespace.copy()
        unique_x1 = np.unique(x_total_ctrl[:, 0])
        unique_x2 = np.unique(x_total_ctrl[:, 1])

        # Determine if we have covariates and get covariate range
        if len(self.bo_covariates) > 0 and not df_results.empty:
            cov_vals = df_results[self.bo_covariates[0].name].values
            cov_min = float(np.nanmin(cov_vals))
            cov_max = float(np.nanmax(cov_vals))
            cov_mean = float(np.nanmean(cov_vals))
            n_cov_samples = 10

            # Sample covariates for marginalization
            cov_samples = np.linspace(cov_min, cov_max, n_cov_samples)
        else:
            cov_mean = 0.0
            cov_samples = np.array([0.0])
            n_cov_samples = 1

        # Create grid for predictions and acquisition (with covariate samples)
        x_grid_full = []
        for x1 in unique_x1:
            for x2 in unique_x2:
                for cov in cov_samples:
                    x_grid_full.append(
                        [x1, x2, cov] if len(self.bo_covariates) > 0 else [x1, x2]
                    )
        x_grid_full = np.array(x_grid_full)

        # Scale and predict
        x_grid_scaled = x_scaler.transform(x_grid_full)
        y_pred_scaled, y_samples_scaled = gp_model.predict(
            rng_key_predict, x_grid_scaled, noiseless=True
        )
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

        # Compute prediction uncertainty (std) in original scale
        y_std_scaled = jnp.std(y_samples_scaled, axis=(0, 1))
        y_std = np.asarray(y_std_scaled) * float(jnp.abs(y_scaler.std_[0]))

        # Marginalize predictions and acquisition over covariate
        if len(self.bo_covariates) > 0:
            # Reshape: (n_x1 * n_x2, n_cov_samples)
            n_ctrl_points = len(unique_x1) * len(unique_x2)
            y_pred_reshaped = y_pred.reshape(n_ctrl_points, n_cov_samples)
            y_pred_marg = y_pred_reshaped.mean(axis=1)

            y_std_reshaped = y_std.reshape(n_ctrl_points, n_cov_samples)
            y_std_marg = y_std_reshaped.mean(axis=1)

            # acq_values_total is already marginalized from _determine_next_parameters
            acq_marg = np.asarray(acq_values_total)
        else:
            y_pred_marg = y_pred
            y_std_marg = y_std
            acq_marg = np.asarray(acq_values_total)

        # Reshape for 2D plotting - meshgrid uses 'ij' indexing, so we need (n_x1, n_x2)
        y_pred_2d = y_pred_marg.reshape(len(unique_x1), len(unique_x2))
        acq_2d = acq_marg.reshape(len(unique_x1), len(unique_x2))

        # Create meshgrid for proper pcolormesh plotting
        X_mesh, Y_mesh = np.meshgrid(unique_x1, unique_x2, indexing="ij")

        # Compute ground truth (None if not available)
        y_true = self._get_ground_truth_grid(unique_x1, unique_x2, cov_mean)

        # Determine common y-axis limits across all plots
        all_y_values = []
        if not df_results.empty:
            all_y_values.append(df_results[self.objective_metric.name].values)
        all_y_values.append(y_pred_marg)
        if y_true is not None:
            all_y_values.append(y_true.flatten())
        all_y_values = np.concatenate(all_y_values)
        y_vmin = float(np.nanmin(all_y_values))
        y_vmax = float(np.nanmax(all_y_values))

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(
            f"BO Iteration {self.iteration + 1}/{self.n_iterations} "
            f"({len(df_results)} total measurements)",
            fontsize=14,
        )

        # Subplot 1: Measured points with covariate as text
        ax1 = axes[0, 0]
        if not df_results.empty:
            scatter = ax1.scatter(
                df_results[self.parameters_to_optimize[0].name],
                df_results[self.parameters_to_optimize[1].name],
                c=df_results[self.objective_metric.name],
                cmap="viridis",
                vmin=y_vmin,
                vmax=y_vmax,
                s=100,
                edgecolor="k",
                linewidth=1.5,
                zorder=3,
            )

            # Add covariate as text annotations if available
            if len(self.bo_covariates) > 0:
                # Group by (x1, x2) and show mean covariate
                grouped = df_results.groupby(
                    [
                        self.parameters_to_optimize[0].name,
                        self.parameters_to_optimize[1].name,
                    ]
                )
                for (x1, x2), group in grouped:
                    cov_val = group[self.bo_covariates[0].name].mean()
                    ax1.annotate(
                        f"{cov_val:.2f}",
                        xy=(x1, x2),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor="white", alpha=0.7
                        ),
                    )

        ax1.set_xlabel(self.parameters_to_optimize[0].name)
        ax1.set_ylabel(self.parameters_to_optimize[1].name)
        ax1.set_title("Measured Points\n(covariate values shown as text)")
        ax1.grid(True, alpha=0.3)

        cov_label = (
            " (marginalized over covariate)" if len(self.bo_covariates) > 0 else ""
        )

        # Subplot 2: Ground Truth (if available) or GP Model Uncertainty
        ax2 = axes[0, 1]
        if y_true is not None:
            im2 = ax2.pcolormesh(
                X_mesh,
                Y_mesh,
                y_true,
                cmap="viridis",
                vmin=y_vmin,
                vmax=y_vmax,
                shading="nearest",
            )
            opt_idx = np.unravel_index(
                (
                    y_true.argmax()
                    if self.objective_metric.goal == "maximize"
                    else y_true.argmin()
                ),
                y_true.shape,
            )
            ax2.scatter(
                unique_x1[opt_idx[0]],
                unique_x2[opt_idx[1]],
                c="red",
                s=300,
                marker="*",
                edgecolors="black",
                linewidths=2,
                label="True optimum",
                zorder=3,
            )
            if not df_results.empty:
                ax2.scatter(
                    df_results[self.parameters_to_optimize[0].name],
                    df_results[self.parameters_to_optimize[1].name],
                    c="white",
                    s=50,
                    marker="x",
                    linewidths=2,
                    label="Measured",
                    zorder=3,
                    alpha=0.8,
                )
            ax2.legend()
            ax2.set_title(f"Ground Truth (cov={cov_mean:.2f})")
        else:
            y_std_2d = y_std_marg.reshape(len(unique_x1), len(unique_x2))
            im2 = ax2.pcolormesh(
                X_mesh,
                Y_mesh,
                y_std_2d,
                cmap="YlOrRd",
                shading="nearest",
            )
            fig.colorbar(im2, ax=ax2, label=f"std({self.objective_metric.name})")
            if not df_results.empty:
                ax2.scatter(
                    df_results[self.parameters_to_optimize[0].name],
                    df_results[self.parameters_to_optimize[1].name],
                    c="blue",
                    s=50,
                    marker="x",
                    linewidths=2,
                    label="Measured",
                    zorder=3,
                )
                ax2.legend()
            ax2.set_title(f"GP Model Uncertainty{cov_label}")
        ax2.set_xlabel(self.parameters_to_optimize[0].name)
        ax2.set_ylabel(self.parameters_to_optimize[1].name)
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Acquisition Function (marginalized)
        ax3 = axes[1, 0]
        # Normalize acquisition for better visualization
        # acq_norm = (acq_2d - acq_2d.min()) / (acq_2d.max() - acq_2d.min() + 1e-10)
        im3 = ax3.pcolormesh(X_mesh, Y_mesh, acq_2d, cmap="coolwarm", shading="nearest")

        # Mark already measured points
        if (
            self.x_performed_experiments is not None
            and len(self.x_performed_experiments) > 0
        ):
            ax3.scatter(
                self.x_performed_experiments[:, 0],
                self.x_performed_experiments[:, 1],
                c="gray",
                s=80,
                marker="x",
                linewidths=2,
                label="Already measured",
                alpha=0.6,
                zorder=2,
            )

        # Mark the point with highest acquisition among unmeasured points
        # Use x_unmeasured_at_computation if provided, otherwise fall back to self.x_unmeasured
        x_unmeasured_for_plot = (
            x_unmeasured_at_computation
            if x_unmeasured_at_computation is not None
            else self.x_unmeasured
        )

        if x_unmeasured_for_plot is not None and len(x_unmeasured_for_plot) > 0:
            x1_to_i = {v: i for i, v in enumerate(unique_x1)}
            x2_to_j = {v: j for j, v in enumerate(unique_x2)}
            mask = np.zeros_like(acq_2d, dtype=bool)
            for x1_u, x2_u in x_unmeasured_for_plot:
                i = x1_to_i.get(x1_u)
                j = x2_to_j.get(x2_u)
                if i is not None and j is not None:
                    mask[i, j] = True
            if np.any(mask):
                acq_masked = np.where(mask, acq_2d, -np.inf)
                max_acq_idx = np.unravel_index(
                    np.nanargmax(acq_masked), acq_masked.shape
                )
            else:
                max_acq_idx = np.unravel_index(acq_2d.argmax(), acq_2d.shape)
        else:
            max_acq_idx = np.unravel_index(acq_2d.argmax(), acq_2d.shape)

        ax3.scatter(
            unique_x1[max_acq_idx[0]],
            unique_x2[max_acq_idx[1]],
            c="yellow",
            s=300,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label="Best acq (unmeasured)",
            zorder=3,
        )

        # Explicitly mark the actually chosen next point
        if next_point is not None:
            ax3.scatter(
                float(next_point[0]),
                float(next_point[1]),
                c="cyan",
                s=220,
                marker="X",
                edgecolors="black",
                linewidths=1.5,
                label="Chosen next",
                zorder=4,
            )
        ax3.legend()
        ax3.set_xlabel(self.parameters_to_optimize[0].name)
        ax3.set_ylabel(self.parameters_to_optimize[1].name)
        acq_label = getattr(
            self, "_acquisition_used_this_round", self.acquisition_function
        )
        ax3.set_title(f"Acquisition Function ({str(acq_label).upper()}){cov_label}")
        ax3.grid(True, alpha=0.3)

        # Subplot 4: GP Predicted Landscape (always)
        ax4 = axes[1, 1]
        im4 = ax4.pcolormesh(
            X_mesh,
            Y_mesh,
            y_pred_2d,
            cmap="viridis",
            vmin=y_vmin,
            vmax=y_vmax,
            shading="nearest",
        )
        # Mark GP-predicted optimum
        gp_opt_idx = np.unravel_index(
            (
                y_pred_2d.argmax()
                if self.objective_metric.goal == "maximize"
                else y_pred_2d.argmin()
            ),
            y_pred_2d.shape,
        )
        ax4.scatter(
            unique_x1[gp_opt_idx[0]],
            unique_x2[gp_opt_idx[1]],
            c="red",
            s=300,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label="GP-predicted optimum",
            zorder=3,
        )
        # Overlay measured points
        if not df_results.empty:
            ax4.scatter(
                df_results[self.parameters_to_optimize[0].name],
                df_results[self.parameters_to_optimize[1].name],
                c="white",
                s=50,
                marker="x",
                linewidths=2,
                label="Measured",
                zorder=3,
                alpha=0.8,
            )
        ax4.legend()
        ax4.set_xlabel(self.parameters_to_optimize[0].name)
        ax4.set_ylabel(self.parameters_to_optimize[1].name)
        ax4.set_title(f"GP Predicted Landscape{cov_label}")
        ax4.grid(True, alpha=0.3)

        # Add some spacing between subplots before adding colorbar
        plt.subplots_adjust(right=0.92)

        # Shared colorbar for subplots using viridis y-value scale (1, 2/gt, 4)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        fig.colorbar(
            scatter if not df_results.empty else im4,
            cax=cbar_ax,
            label=self.objective_metric.name,
        )

        plt.show()

    @abstractmethod
    def _create_df_acquire_for_exp_cycle(self, parameters: dict) -> pd.DataFrame:
        pass

    @abstractmethod
    def _run_experiment_and_preprocess_results(
        self, df_acquire: pd.DataFrame
    ) -> pd.DataFrame:
        # needs to be implemented by user to run experiment based on df_acquire and preprocess results into df_results format
        # Implementation of running experiment and preprocessing results
        pass

    def run(self):
        df_results = pd.DataFrame()

        initial_parameters, _ = self._select_initial_samples(k=3)

        # _select_initial_samples returns a list of parameter dicts.
        # Keep _create_df_acquire_for_exp_cycle API as a single-point call
        # and concatenate the initial acquisitions.
        if isinstance(initial_parameters, dict):
            initial_parameters = [initial_parameters]

        for i, params in enumerate(initial_parameters):
            print(f"=== Initial sample {i + 1}/{len(initial_parameters)}: {params} ===")
            df_acquisition = self._create_df_acquire_for_exp_cycle(params)
            df_new_results = self._run_experiment_and_preprocess_results(df_acquisition)
            df_results = pd.concat([df_results, df_new_results], ignore_index=True)
            if not df_new_results.empty:
                self._iteration_means.append(
                    float(df_new_results[self.objective_metric.name].mean())
                )

        for e in range(self.n_iterations):
            print(f"\n=== BO Iteration {e + 1}/{self.n_iterations} ===")
            next_params = self._determine_next_parameters(
                df_results, verbose=self.verbose
            )
            print(f"Selected parameters for next experiment: {next_params}")
            new_df_acquire = self._create_df_acquire_for_exp_cycle(next_params)
            new_df_results = self._run_experiment_and_preprocess_results(new_df_acquire)
            df_results = pd.concat([df_results, new_df_results], ignore_index=True)
            if not new_df_results.empty:
                self._iteration_means.append(
                    float(new_df_results[self.objective_metric.name].mean())
                )
            self.iteration += 1

        # Store final results for external access
        if not df_results.empty:
            self.x = self._extract_x_from_df(df_results)
            self.y = self._extract_y_from_df(df_results)
        self.df_results = df_results


# Module-level RNG so repeated calls to mock_df_results don't reset covariate samples.
# This keeps runs reproducible while allowing covariate values to vary across BO rounds.
_MOCK_RNG = np.random.default_rng(42)


def mock_df_results(
    df_acquire: pd.DataFrame,
    n_samples: int = 10,
    *,
    noise_scale: float = 1.0,
    noise_base: float = 2.0,
    noise_cov_scale: float = 2.0,
) -> pd.DataFrame:
    """Generate multiple measurements for each df_acquire row.

    The objective is maximized around a moving optimum influenced by cov1.
    """
    rng = _MOCK_RNG
    rows = []
    for _, row in df_acquire.iterrows():
        x1 = float(row["x1"])
        x2 = float(row["x2"])

        # Non-steerable covariate fluctuates per sample.
        cov1_samples = rng.uniform(0.0, 1.0, size=n_samples)

        # Covariate shifts the optimum slightly.
        x1_opt = 6.0 + 1.5 * (cov1_samples - 0.5)
        x2_opt = 2.5 - 1.0 * (cov1_samples - 0.5)

        # Smooth objective landscape (negative quadratic bowl + sinusoidal interaction).
        base = 200.0
        quad = -2.0 * (x1 - x1_opt) ** 2 - 3.0 * (x2 - x2_opt) ** 2
        interaction = 6.0 * np.sin(0.5 * x1) * np.cos(0.7 * x2)

        # Noise increases slightly with covariate magnitude.
        # `noise_scale` is the main knob: 0.0 -> deterministic, 1.0 -> default.
        noise_std = noise_scale * (noise_base + noise_cov_scale * cov1_samples)
        noise = rng.normal(0.0, noise_std, size=n_samples)

        y_samples = base + quad + interaction + 10.0 * cov1_samples + noise

        for i in range(n_samples):
            rows.append(
                {
                    "x1": x1,
                    "x2": x2,
                    "cov1": cov1_samples[i],
                    "y": y_samples[i],
                }
            )

    return pd.DataFrame(rows)


def compute_ground_truth(x1_grid, x2_grid, cov1_mean):
    """Compute ground truth function values on a grid.

    Returns array with shape (len(x1_grid), len(x2_grid)) using 'ij' indexing.
    """
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid, indexing="ij")

    # Covariate shifts the optimum slightly.
    x1_opt = 6.0 + 1.5 * (cov1_mean - 0.5)
    x2_opt = 2.5 - 1.0 * (cov1_mean - 0.5)

    # Smooth objective landscape (negative quadratic bowl + sinusoidal interaction).
    base = 200.0
    quad = -2.0 * (x1_mesh - x1_opt) ** 2 - 3.0 * (x2_mesh - x2_opt) ** 2
    interaction = 6.0 * np.sin(0.5 * x1_mesh) * np.cos(0.7 * x2_mesh)

    y_true = base + quad + interaction + 10.0 * cov1_mean

    return y_true


def compute_protrusion_ground_truth(
    angle_grid,
    radial_grid,
    opto_rtk_mean,
    *,
    n_stim_frames=10,
    cell_radius=20.0,
    patch_radius=10,
    n_vertices=24,
    protrusion_gain=0.03,
):
    """Physics-based ground truth for the protrusion simulation.

    Models the geometric intersection of a circular stimulation patch
    (radius ``patch_radius``) with the boundary vertices of a roughly
    circular cell (radius ``cell_radius``, ``n_vertices`` vertices).

    The patch centre is placed at distance ``stim_radial * cell_radius``
    from the cell centre.  A boundary vertex at angular offset *phi*
    from the patch direction is hit when::

        R^2 + d^2 - 2 R d cos(phi) < patch_radius^2

    Because the cell is approximately circular, ``stim_angle`` has
    negligible effect -- the ground-truth surface is essentially
    one-dimensional (depends only on ``stim_radial``).

    Returns array with shape ``(len(angle_grid), len(radial_grid))``
    using ``'ij'`` indexing.
    """
    _sa, sr = np.meshgrid(angle_grid, radial_grid, indexing="ij")

    R = cell_radius
    p = patch_radius
    d = sr * R  # distance from cell centre to patch centre

    # -- Fraction of boundary arc covered by the stimulation patch ------
    # u = cos(half-angle) threshold; arc is hit where cos(phi) > u.
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(d > 1e-6, (R**2 + d**2 - p**2) / (2 * R * d), np.inf)

    arc_frac = np.where(
        u >= 1.0,
        0.0,
        np.where(u <= -1.0, 1.0, np.arccos(np.clip(u, -1, 1)) / np.pi),
    )

    # -- Per-frame fractional area change --------------------------------
    # effective_gain = protrusion_gain * position_factor * opto_rtk
    # position_factor ~ 1.0 (hit vertices sit on the boundary at distance R)
    # Each hit vertex adds gain*R to its radius; the corresponding area
    # wedge grows by ~2*(arc_frac)*gain*opto_rtk relative to total area.
    per_frame = 2.0 * arc_frac * protrusion_gain * opto_rtk_mean

    # Compound over stimulation frames.
    y_true = (1.0 + per_frame) ** n_stim_frames - 1.0

    return y_true
