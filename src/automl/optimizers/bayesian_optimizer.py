import numpy as np
import time
import logging
from automl.optimizers.base_optimizer import BaseOptimizer
from automl.optimizers.surrogate_models import GaussianProcessSurrogate
from automl.optimizers.acquisition_functions import ExpectedImprovement
from sklearn.preprocessing import LabelEncoder
from automl.config import get_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, task, time_budget, surrogate_model=GaussianProcessSurrogate(), acquisition_function=ExpectedImprovement(), verbose=False):
        """
        Parameters:
            task (str): "classification" or "regression"
            time_budget (int): optimization time in seconds
            surrogate_model: an object that supports .fit(X, y) and is used by the acquisition function
            acquisition_function: an object with a method .evaluate(candidates, surrogate_model, best_score)
            config (dict): shared configuration containing default_metric and models definitions.
            verbose (bool): whether to print detailed logging (default False)
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
         - Randomly choose a model (categorical, encoded numerically)
         - For the chosen model, sample one value for each hyperparameter.
        """
        candidate = {}
        model_names = list(self.state_space.keys())
        chosen_model = np.random.choice(model_names)
        candidate["model"] = self.model_encoder.transform([chosen_model])[0]
        for param, values in self.state_space[chosen_model].items():
            candidate[param] = np.random.choice(values)
        return candidate

    def _build_model(self, candidate_params: dict):
        """
        Decode candidate parameters and instantiate a model.
        """
        decoded_model = self.model_encoder.inverse_transform([candidate_params["model"]])[0]
        candidate = candidate_params.copy()
        candidate["model"] = decoded_model

        if decoded_model.lower() == "randomforest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=candidate.get("n_estimators"),
                max_depth=candidate.get("max_depth"),
                min_samples_split=candidate.get("min_samples_split")
            )
        elif decoded_model.lower() == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                learning_rate=candidate.get("learning_rate"),
                n_estimators=candidate.get("n_estimators"),
                max_depth=candidate.get("max_depth"),
                gamma=candidate.get("gamma")
            )
        elif decoded_model.lower() == "logisticregression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=candidate.get("C"))
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
        return model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Perform Bayesian Optimization to find the best hyperparameter configuration.
        Uses a fixed candidate vector (all_keys) to ensure homogeneous input for the surrogate model.
        Returns the final fitted model.
        """
        history = []
        start_time = time.time()
        logger.info("Starting Bayesian optimization with a time budget of %d seconds", self.time_budget)

        # --- Initial Random Sampling ---
        num_initial_samples = 5
        for i in range(num_initial_samples):
            candidate = self._generate_candidate()
            score = self.evaluate_candidate(self._build_model, candidate, X, y)
            history.append((candidate, score))
            if score > self.best_score:
                self.best_score = score
            logger.info("Initial sample %d: Candidate %s yielded score %.4f", i + 1, candidate, score)

        # --- Establish a Fixed Order for Candidate Parameters ---
        # "model" is always first, then all hyperparameter keys (sorted for consistency)
        all_keys = ["model"] + sorted({key for model in self.state_space.values() for key in model.keys()})
        logger.info("Using candidate keys: %s", all_keys)

        # --- Main Bayesian Optimization Loop ---
        while time.time() - start_time < self.time_budget:
            # Convert history candidates to homogeneous numeric vectors.
            # For each candidate, use the fixed order; if a key is missing, fill with -1.
            X_history = np.array(
                [[cand.get(key, -1) for key in all_keys] for cand, _ in history],
                dtype=np.float64
            )
            y_history = np.array([score for _, score in history])

            # (If necessary, transform the "model" field.
            # In our design, candidates generated by _generate_candidate already encode the model
            # as a number; if not, you might need:
            # X_history[:, 0] = self.model_encoder.transform(X_history[:, 0])
            # )

            self.surrogate_model.fit(X_history, y_history)

            # Generate new candidates.
            candidates = [self._generate_candidate() for _ in range(30)]
            # Convert candidates to a homogeneous array using the same fixed key order.
            candidates_vec = np.array(
                [[cand.get(key, -1) for key in all_keys] for cand in candidates],
                dtype=np.float64
            )
            acquisition_values = self.acquisition_function.evaluate(candidates_vec, self.surrogate_model,
                                                                    self.best_score)
            best_candidate = candidates[np.argmax(acquisition_values)]
            score = self.evaluate_candidate(self._build_model, best_candidate, X, y)
            history.append((best_candidate, score))
            if score > self.best_score:
                self.best_score = score
            remaining_time = self.time_budget - (time.time() - start_time)
            logger.info("Evaluated candidate %s yielded score %.4f; Best score: %.4f; Remaining time: %.2fs",
                        best_candidate, score, self.best_score, remaining_time)

        # --- Final Selection and Model Fitting ---
        best_params = max(history, key=lambda item: item[1])[0]
        logger.info("Best candidate found: %s", best_params)
        final_model = self._build_model(best_params)
        final_model.fit(X, y)
        self.optimal_model = final_model
        self.metric_value = self.best_score
        logger.info("Optimization complete. Best model: %s with score %.4f",
                    final_model.__class__.__name__, self.best_score)
        return final_model

