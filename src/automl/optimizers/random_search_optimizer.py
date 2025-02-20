import numpy as np
import pandas as pd
import time
import logging
from automl.optimizers.base_optimizer import BaseOptimizer
from automl.config import get_config
from automl.enums import Task
import numpy as np
import time
import logging
from automl.optimizers.surrogate_models import GaussianProcessSurrogate
from automl.optimizers.acquisition_functions import ExpectedImprovement
from sklearn.preprocessing import LabelEncoder
from automl.config import get_config
from automl.enums import Task
import pandas as pd
from celery import shared_task
from automl.enums import Task, Metric
from automl.functions import createPipeline
import joblib
import os
import pandas as pd


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
        self.best_score = -np.inf if self.metric != "neg_mean_squared_error" else np.inf  # Lower is better for MSE

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

    def _build_model(self, candidate_params: dict):
        model_name = candidate_params["model"]

        if model_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            ModelClass = RandomForestClassifier if self.task == Task.CLASSIFICATION else RandomForestRegressor
            model = ModelClass(
                n_estimators=candidate_params.get("n_estimators"),
                max_depth=candidate_params.get("max_depth"),
                min_samples_split=candidate_params.get("min_samples_split"),
            )

        elif model_name == "XGBoost":
            from xgboost import XGBClassifier, XGBRegressor
            ModelClass = XGBClassifier if self.task == Task.CLASSIFICATION else XGBRegressor
            model = ModelClass(
                learning_rate=candidate_params.get("learning_rate"),
                n_estimators=candidate_params.get("n_estimators"),
                max_depth=candidate_params.get("max_depth"),
                gamma=candidate_params.get("gamma"),
            )

        elif model_name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=candidate_params.get("C"))

        elif model_name == "LinearRegression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

        elif model_name == "Ridge":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=candidate_params.get("alpha"))

        elif model_name == "Lasso":
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=candidate_params.get("alpha"))

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Perform random search to find the best hyperparameter configuration.

        Returns:
            Trained model with the best found parameters.
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

        final_model = self._build_model(best_params)
        final_model.fit(X, y)
        self.optimal_model = final_model
        self.metric_value = self.best_score

        logger.info("Optimization complete. Best model: %s with score %.4f",
                    final_model.__class__.__name__, self.best_score)
        return final_model
    
if __name__ == '__main__':
    df = pd.read_csv(r'D:\1GRADUATION-PROJECT\Protoype\prototype-backend\automlapp\tests\diabetes.csv')
    print(df)
    pl = createPipeline(df, 'Outcome')
    df = pl.transform(df)
    hpo = RandomSearchOptimizer(task = Task.CLASSIFICATION, time_budget=10)

    # hpo = bayesian_optimizer_hyperband(task = Task.CLASSIFICATION, time_budget=150)
    X = df.drop(columns=["Outcome"])
    Y = df['Outcome']
    hpo.fit(X,Y)
    accuracy = hpo.get_metric_value()
    print(accuracy)
    
