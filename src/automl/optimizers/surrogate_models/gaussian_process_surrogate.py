from .base_surrogate_model import BaseSurrogateModel
from typing import Tuple, Any
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from automl.enums import Task


class GaussianProcessSurrogate(BaseSurrogateModel):
    def __init__(self, task: Task):
        """
        A unified Gaussian Process surrogate that branches to GPR or GPC:
            - Classification -> GaussianProcessClassifier with Matern kernel.
            - Regression    -> GaussianProcessRegressor with RBF kernel.
        """
        self.task = task

        if self.task == Task.CLASSIFICATION:
            kernel = C(1.0, (1e-3, 1e5)) * Matern(
                length_scale=1.0,
                nu=1.5,
                length_scale_bounds=(1e-3, 1e5)
            )
            self.model = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=20
            )
        else:
            kernel = C(1.0, (1e-3, 1e5)) * RBF(
                length_scale=1.0,
                length_scale_bounds=(1e-3, 1e5)
            )
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-7,
                n_restarts_optimizer=20
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit either a GPC or GPR depending on the task.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray, return_std: bool = False) -> tuple[Any, None] | tuple[Any, Any]:
        """
        - For regression (GPR): returns (mean, std) if return_std=True.
        - For classification (GPC):
            -> predict() returns discrete class labels,
               no built-in standard deviation concept.
               If return_std=True, raises NotImplementedError.
        """
        if self.task == Task.CLASSIFICATION:
            if return_std:
                raise NotImplementedError(
                    "GaussianProcessClassifier doesn't provide std dev directly. "
                    "Use predict_proba or a custom uncertainty measure if needed."
                )
            labels = self.model.predict(X)
            return labels, None
        else:
            # Regression
            if return_std:
                y_mean, y_std = self.model.predict(X, return_std=True)
                return y_mean, y_std
            else:
                y_mean = self.model.predict(X, return_std=False)
                return y_mean, None
