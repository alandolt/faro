from rtm_pymmcore.agents.abstract_agent import Agent
import pandas as pd
import dataclasses
from typing import List
import gpax
import jax
import jax.numpy as jnp
import numpy as np
import gpax.utils
from gpax.models.gp import ExactGP

from gpax import acquisition
import gpax.utils


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
        pipeline,
        microscope,
        parameters_to_optimize: List[BO_Parameter],
        objective_metric: BO_Objective,
        bo_covariates: List[BO_Covariate] = [],
        n_iterations: int = 10,
        prior=None,
    ):

        self.n_iterations = n_iterations
        self.iteration = 0
        self.microscope = microscope
        self.pipeline = pipeline
        self.parameters_to_optimize = parameters_to_optimize
        self.total_parameter_space = self._generate_total_parameter_space()
        self.objective_metric = objective_metric
        self.n_iterations = n_iterations
        self.model = None  # Placeholder for the GP model
        self.fovs = None
        self.measured_params = None
        self.x = None
        self.y = None
        self.x_covariate = None
        self._seed = 42

        self.x_unmeasured = self._generate_total_parameter_space()
        self.bo_covariates = bo_covariates

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
        return X_selected, selected_indices

    # def _determine_next_parameters(self, df_results: pd.DataFrame) -> dict:

    #     x = self._extract_x_from_df(df_results)
    #     y = self._extract_y_from_df(df_results)

    #     x_scaler = StandardScalerBounds()
    #     bounds, log_scale = self._get_bounds_and_log_scale()
    #     x = x_scaler.fit_transform(x, bounds=bounds, log_scale=log_scale)

    #     self.x = np.concatenate([self.x, x], axis=0) if self.x is not None else x
    #     self.y = np.concatenate([self.y, y], axis=0) if self.y is not None else y

    #     rng_key, rng_key_predict = gpax.utils.get_keys()

    #     gp_model = gpax.ExactGP(kernel="Matern", input_dim=self.x.shape[1])

    #     gp_model.fit(rng_key, X=self.x, y=self.y, progress_bar=True)

    #     y_pred, y_sampled = gp_model.predict(rng_key_predict, X_unmeasured, noiseless=True)
    #     # Compute acquisition function for each candidate point
    #     # Note: We compute qUCB for all candidates, then marginalize over expression
    #     acq_values = acquisition.qUCB(
    #         rng_key_predict,
    #         gp_model,
    #         X_unmeasured,
    #         beta=4.0,
    #         maximize=True,
    #         noiseless=True,
    #         # maximize_distance=True
    #     )

    #     return {}  # Placeholder implementation

    def _create_df_acquire_for_exp_cycle(self, parameters: dict) -> pd.DataFrame:
        # Implementation of creating df_acquire for experiment cycle
        return pd.DataFrame()  # Placeholder implementation

    def run(self):
        df_results = pd.DataFrame()
        for _ in range(self.n_iterations):
            next_params = self._determine_next_parameters(df_results)
            df_acquire = self._create_df_acquire_for_exp_cycle(next_params)
            self.microscope.run_experiment(df_acquire)
            df_new_results = self.pipeline.get_results_dataframe()
            df_results = pd.concat([df_results, df_new_results], ignore_index=True)
            self.iteration += 1
