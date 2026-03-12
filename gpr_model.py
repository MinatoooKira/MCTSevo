"""Gaussian Process Regression model for fitness prediction."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


class FitnessGPR:
    """Wraps a sklearn GPR that maps ESM-2 embeddings → predicted fitness."""

    def __init__(self):
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._gpr: Optional[GaussianProcessRegressor] = None
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        self._trained = False

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def n_samples(self) -> int:
        return 0 if self._y is None else len(self._y)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """(Re)train the GPR from scratch on all accumulated data.

        Parameters
        ----------
        X : (N, D) embeddings
        y : (N,) fitness values
        """
        if len(X) < 2:
            print("[GPR] Need at least 2 samples to train – skipping.")
            return

        self._X = X.copy()
        self._y = y.copy()

        X_scaled = self._scaler_X.fit_transform(X)
        y_scaled = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=False,
        )
        self._gpr.fit(X_scaled, y_scaled)
        self._trained = True
        print(f"[GPR] Trained on {len(y)} samples. Kernel: {self._gpr.kernel_}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict fitness (and uncertainty) for embedding(s).

        Parameters
        ----------
        X : (D,) or (N, D) embeddings

        Returns
        -------
        mean : predicted fitness in original scale
        std  : predictive standard deviation in original scale
        """
        if not self._trained:
            X = np.atleast_2d(X)
            return np.zeros(X.shape[0]), np.ones(X.shape[0])

        X = np.atleast_2d(X)
        X_scaled = self._scaler_X.transform(X)
        mean_scaled, std_scaled = self._gpr.predict(X_scaled, return_std=True)

        y_scale = self._scaler_y.scale_[0] if self._scaler_y.scale_ is not None else 1.0
        y_mean = self._scaler_y.mean_[0] if self._scaler_y.mean_ is not None else 0.0

        mean = mean_scaled * y_scale + y_mean
        std = std_scaled * y_scale
        return mean, std
