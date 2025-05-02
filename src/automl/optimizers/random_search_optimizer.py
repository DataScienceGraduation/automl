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
            task (str): "classification", "regression", or "clustering"
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

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Perform random search to find the best hyperparameter configuration.
        Returns a trained model with those parameters.
        
        Parameters:
            X: np.ndarray - Feature matrix
            y: np.ndarray - Target values (not used for clustering)
        """
        history = []
        start_time = time.time()
        logger.info("Starting Random Search optimization with a time budget of %d seconds", self.time_budget)

        while time.time() - start_time < self.time_budget:
            candidate = self._generate_candidate()
            score = self.evaluate_candidate(self.build_model, candidate, X, y)
            history.append((candidate, score))

            if self.task == Task.CLUSTERING:
                # For clustering, higher silhouette score is better
                if score > self.best_score:
                    self.best_score = score
            elif self.task == Task.REGRESSION:
                # For regression, lower RMSE is better
                if score < self.best_score:
                    self.best_score = score
            else:
                # For classification, higher accuracy is better
                if score > self.best_score:
                    self.best_score = score

            remaining_time = self.time_budget - (time.time() - start_time)
            logger.info("Candidate: %s yielded score: %.4f | Best score: %.4f | Time left: %.2fs",
                        candidate, score, self.best_score, remaining_time)

        best_params = max(history, key=lambda item: item[1])[0] if self.task != Task.REGRESSION else min(history, key=lambda item: item[1])[0]
        logger.info("Best candidate found: %s", best_params)

        final_model = self.build_model(best_params)
        final_model.fit(X)

        self.optimal_model = final_model
        self.metric_value = self.best_score

        logger.info("Optimization complete. Best model: %s with score %.4f",
                    final_model.__class__.__name__, self.best_score)
        return final_model
    
if __name__ == '__main__':
    df = pd.read_csv(r'D:\College\Grad\AutoML\data_clustering\fish_data.csv')
    print(df.head())
    pl = createPipeline(df, task="clustering")
    df = pl.transform(df)
    hpo = RandomSearchOptimizer(task = Task.CLUSTERING, time_budget=300)

    # hpo = bayesian_optimizer_hyperband(task = Task.CLASSIFICATION, time_budget=150)
    X = df.drop(columns=["species"])
    # Y = df['Outcome']
    hpo.fit(X)
    accuracy = hpo.get_metric_value()
    print(accuracy)
