import numpy as np
import time
import logging
from automl.optimizers.base_optimizer import BaseOptimizer
from automl.optimizers.surrogate_models import GaussianProcessSurrogate
from automl.optimizers.acquisition_functions import ExpectedImprovement
from sklearn.preprocessing import LabelEncoder
from automl.config import get_config
from automl.enums import Task

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, task, time_budget,
                 surrogate_model=GaussianProcessSurrogate(task=Task.REGRESSION),
                 acquisition_function=ExpectedImprovement(),
                 verbose=False):
        """
        Parameters:
            task (str): "classification" or "regression"
            time_budget (int): optimization time in seconds
            surrogate_model: an object that supports .fit(X, y)
            acquisition_function: an object that implements .evaluate(...)
            verbose (bool): whether to log detailed info (default False)
        """
        config = get_config(task)
        super().__init__(task, time_budget, metric=None, verbose=verbose, config=config)

        self.surrogate_model = surrogate_model
        self.acquisition_function = acquisition_function

        self.state_space = self.models_config

        self.model_encoder = LabelEncoder()
        model_names = list(self.state_space.keys())
        self.model_encoder.fit(model_names)

        self.best_score = -np.inf

    def _generate_candidate(self) -> dict:
        """
        Generate a candidate configuration:
         - Randomly choose a model (numerically encoded)
         - For that chosen model, sample one value for each hyperparameter.
        """
        candidate = {}
        model_names = list(self.state_space.keys())
        chosen_model = np.random.choice(model_names)
        candidate["model"] = self.model_encoder.transform([chosen_model])[0]

        for param, values in self.state_space[chosen_model].items():
            candidate[param] = np.random.choice(values)

        return candidate

    def build_model(self, candidate_params: dict):
        """
        Decode the numeric 'model' field back to a string, then
        delegate actual model creation to the BaseOptimizer's method.
        """
        candidate = candidate_params.copy()
        decoded_model = self.model_encoder.inverse_transform([candidate["model"]])[0]
        candidate["model"] = decoded_model
        return super().build_model(candidate)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Bayesian Optimization loop:
         1) Random initial samples (some random candidates)
         2) Fit the surrogate model on (X_history, y_history)
         3) Evaluate acquisition function => pick best candidate
         4) Evaluate candidate => update best_score
         5) Repeat until time_budget is reached
        """
        history = []
        start_time = time.time()
        logger.info("Starting Bayesian optimization with a time budget of %d seconds", self.time_budget)

        num_initial_samples = 5
        for i in range(num_initial_samples):
            candidate = self._generate_candidate()
            score = self.evaluate_candidate(self.build_model, candidate, X, y)
            history.append((candidate, score))

            if score > self.best_score:
                self.best_score = score

            logger.info("Initial sample %d: Candidate %s yielded score %.4f",
                        i + 1, candidate, score)

        all_keys = ["model"] + sorted({key for model_dict in self.state_space.values() for key in model_dict.keys()})
        logger.info("Using candidate keys: %s", all_keys)

        while time.time() - start_time < self.time_budget:
            X_history = np.array(
                [[cand.get(key, -1) for key in all_keys] for cand, _ in history],
                dtype=np.float64
            )
            y_history = np.array([score for _, score in history])

            self.surrogate_model.fit(X_history, y_history)

            candidates = [self._generate_candidate() for _ in range(30)]
            candidates_vec = np.array(
                [[cand.get(key, -1) for key in all_keys] for cand in candidates],
                dtype=np.float64
            )

            acquisition_values = self.acquisition_function.evaluate(
                candidates_vec, self.surrogate_model, self.best_score
            )
            best_candidate = candidates[np.argmax(acquisition_values)]

            score = self.evaluate_candidate(self.build_model, best_candidate, X, y)
            history.append((best_candidate, score))

            if score > self.best_score:
                self.best_score = score

            remaining_time = self.time_budget - (time.time() - start_time)
            logger.info("Evaluated candidate %s yielded score %.4f; Best score: %.4f; Remaining time: %.2fs",
                        best_candidate, score, self.best_score, remaining_time)

        best_params = max(history, key=lambda item: item[1])[0]
        logger.info("Best candidate found: %s", best_params)

        final_model = self.build_model(best_params)
        final_model.fit(X, y)

        self.optimal_model = final_model
        self.metric_value = self.best_score

        logger.info("Optimization complete. Best model: %s with score %.4f",
                    final_model.__class__.__name__, self.best_score)
        return final_model