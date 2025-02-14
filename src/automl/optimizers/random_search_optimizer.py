import numpy as np
import time
import logging
from automl.optimizers.base_optimizer import BaseOptimizer
from automl.config import get_config
from automl.enums import Task

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, task, time_budget, verbose=False):
        """
        Random search hyperparameter optimizer.

        Parameters:
            task (str): "classification" or "regression"
            time_budget (int): time limit in seconds for searching optimal hyperparameters
            verbose (bool): whether to log detailed output (default False)
        """
        config = get_config(task)
        super().__init__(task, time_budget, metric=None, verbose=verbose, config=config)
        self.best_score = -np.inf if self.metric != "neg_mean_squared_error" else np.inf

    def _generate_candidate(self) -> dict:
        """
        Randomly generate a candidate hyperparameter configuration.
        """
        candidate = {}
        model_names = list(self.models_config.keys())
        chosen_model = np.random.choice(model_names)
        candidate["model"] = chosen_model

        for param, values in self.models_config[chosen_model].items():
            candidate[param] = np.random.choice(values)
        return candidate

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Perform random search to find the best hyperparameter configuration.
        Returns a trained model with those parameters.
        """
        history = []
        start_time = time.time()
        logger.info("Starting Random Search optimization with a time budget of %d seconds", self.time_budget)

        while time.time() - start_time < self.time_budget:
            candidate = self._generate_candidate()
            score = self.evaluate_candidate(self._build_model, candidate, X, y)
            history.append((candidate, score))

            if (self.metric != "neg_mean_squared_error" and score > self.best_score) or \
                    (self.metric == "neg_mean_squared_error" and score < self.best_score):
                self.best_score = score

            remaining_time = self.time_budget - (time.time() - start_time)
            logger.info("Candidate: %s yielded score: %.4f | Best score: %.4f | Time left: %.2fs",
                        candidate, score, self.best_score, remaining_time)

        best_params = max(history, key=lambda item: item[1])[0]
        logger.info("Best candidate found: %s", best_params)

        final_model = self.build_model(best_params)
        final_model.fit(X, y)

        self.optimal_model = final_model
        self.metric_value = self.best_score

        logger.info("Optimization complete. Best model: %s with score %.4f",
                    final_model.__class__.__name__, self.best_score)
        return final_model
