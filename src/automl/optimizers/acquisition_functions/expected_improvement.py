from .base_acquisition_function import BaseAcquistionFunction
import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExpectedImprovement(BaseAcquistionFunction):
    def evaluate(self, X, surrogate_model, best_observed: float) -> np.ndarray:
        """
        Evaluate the Expected Improvement acquisition function at the given input points.

        Parameters:
            X: Either an iterable of dictionaries (candidate configurations) or
               a NumPy array of shape (n_candidates, n_features) where the candidate
               parameters are already in a fixed order.
            surrogate_model: A fitted surrogate model with a predict method supporting return_std=True.
            best_observed: The best score observed so far.

        Returns:
            ei: An array of expected improvement values.
        """
        logger.info("Evaluating Expected Improvement acquisition function")
        logger.info(f"type of X: {type(X)}")

        if not isinstance(X, np.ndarray):
            try:
                X_arr = np.array([list(x.values()) for x in X])
            except Exception as e:
                logger.error("Error converting candidates to array: %s", e)
                raise
        else:
            X_arr = X

        logger.info("Candidates array shape: %s", X_arr.shape)
        y_pred, sigma = surrogate_model.predict(X_arr, return_std=True)
        best_y = max(y_pred)
        if best_observed > 0.1:
            best_y = best_observed

        with np.errstate(divide='warn'):
            improvement = y_pred - best_y
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        logger.info("Expected Improvement values: %s", ei)
        return ei
