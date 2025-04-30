import numpy as np
import time
import logging
from automl.optimizers.base_optimizer import BaseOptimizer
from automl.optimizers.surrogate_models import GaussianProcessSurrogate
from automl.optimizers.acquisition_functions import ExpectedImprovement
from sklearn.preprocessing import LabelEncoder
from automl.config import get_config
from automl.enums import Task
import pandas as pd
from automl.optimizers import RandomSearchOptimizer
from automl.enums import Task, Metric
from automl.functions import createPipeline
import joblib
import os
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BayesianOptimizerHyperband(BaseOptimizer):
    def __init__(self, task, time_budget, eta=3, surrogate_model=None, acquisition_function=ExpectedImprovement(), verbose=False):
        """
        Parameters:
            task (str): "classification", "regression", "forecasting", or "clustering"
            time_budget (int): total optimization time in seconds
            eta (int): reduction factor in Hyperband (default 3)
            surrogate_model: a model that supports .fit(X, y) and is used by Bayesian Optimization
            acquisition_function: an acquisition function to guide Bayesian search
            verbose (bool): whether to print detailed logging (default False)
        """
        config = get_config(task)
        super().__init__(task, time_budget, metric=None, verbose=verbose, config=config)

        self.surrogate_model = surrogate_model if surrogate_model is not None else GaussianProcessSurrogate(task=task)
        self.acquisition_function = acquisition_function
        self.state_space = self.models_config
        self.eta = eta  # Hyperband reduction factor

        self.model_encoder = LabelEncoder()
        model_names = list(self.state_space.keys())
        self.model_encoder.fit(model_names)

        self.best_score = -np.inf

    def _generate_candidate(self) -> dict:
        """
        Generate a candidate hyperparameter configuration.
        """
        candidate = {}
        model_names = list(self.state_space.keys())
        chosen_model = np.random.choice(model_names)
        candidate["model"] = self.model_encoder.transform([chosen_model])[0]
        for param, values in self.state_space[chosen_model].items():
            candidate[param] = np.random.choice(values)
        return candidate

    def _run_hyperband(self, X, y, R=27):
        """
        Hyperband framework for budget allocation with Bayesian Optimization.
        """
        s_max = int(np.log(R) / np.log(self.eta))  # Number of brackets
        B = (s_max + 1) * R  # Total budget

        for s in reversed(range(s_max + 1)):
            n = int(B / R / (s + 1) * self.eta ** s)  # Number of configurations
            r = R * self.eta ** (-s)  # Initial budget per configuration
            logger.info(f"Hyperband bracket {s}: n={n}, r={r}")
            
            candidates = [self._generate_candidate() for _ in range(n)]
            performances = []
            
            for i, candidate in enumerate(candidates):
                t1 = time.time()
                # Decode the model name for the base class's build_model method
                decoded_candidate = candidate.copy()
                decoded_model = self.model_encoder.inverse_transform([candidate["model"]])[0]
                decoded_candidate["model"] = decoded_model
                
                score = self.evaluate_candidate(self.build_model, decoded_candidate, X, y)
                t2 = time.time()
                T = t2 - t1
                if(T > r):
                    score = -np.inf
                performances.append((candidate, score))
                if score > self.best_score:
                    self.best_score = score
                    self.best_candidate = candidate
                logger.info(f"Candidate {i+1}/{n}: {candidate} -> Score: {score:.4f}")
            
            performances.sort(key=lambda x: x[1], reverse=True)
            candidates = [x[0] for x in performances[:max(1, len(performances) // self.eta)]]
        
        return self.best_candidate

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Perform Bayesian Optimization with Hyperband (BOHB).
        """
        start_time = time.time()
        logger.info("Starting BOHB optimization with a time budget of %d seconds", self.time_budget)
        
        best_params = self._run_hyperband(X, y)
        
        logger.info("Best candidate found: %s", best_params)
        # Decode the model name for the base class's build_model method
        decoded_params = best_params.copy()
        decoded_model = self.model_encoder.inverse_transform([best_params["model"]])[0]
        decoded_params["model"] = decoded_model
        
        final_model = self.build_model(decoded_params)
        final_model.fit(X, y)
        
        self.optimal_model = final_model
        self.metric_value = self.best_score
        logger.info("Optimization complete. Best model: %s with score %.4f", final_model.__class__.__name__, self.best_score)
        return final_model

if __name__ == '__main__':
    df = pd.read_csv(r'D:\1GRADUATION-PROJECT\Protoype\prototype-backend\automlapp\tests\diabetes.csv')
    print(df)
    pl = createPipeline(df, 'Outcome')
    df = pl.transform(df)
    hpo = bayesian_optimizer_hyperband(task = Task.CLASSIFICATION, time_budget=150)

    # hpo = bayesian_optimizer_hyperband(task = Task.CLASSIFICATION, time_budget=150)
    X = df.drop(columns=["Outcome"])
    Y = df['Outcome']
    hpo.fit(X,Y)
    accuracy = hpo.get_metric_value()
    print(accuracy)
    
