from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score
import numpy as np
from automl import Task
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseOptimizer(ABC):
    def __init__(self, task, time_budget, metric=None, verbose=False, cv_folds=10, config=None):
        """
        Parameters:
            task: str - e.g., "classification" or "regression"
            time_budget: int - time allowed for optimization (in seconds)
            metric: str - scoring function (if None, defaults to value in config based on task)
            verbose: bool - whether to print detailed info (default False)
            cv_folds: int - number of cross-validation folds (default 10)
            config: dict - configuration dictionary (e.g., from automl.config)
        """
        self.task = task
        self.time_budget = time_budget
        self.verbose = verbose
        self.cv_folds = cv_folds
        self.config = config or {}

        if metric is None:
            if self.task == Task.CLASSIFICATION:
                self.metric = self.config.get("default_metric", "accuracy")
            elif self.task == Task.REGRESSION:
                self.metric = self.config.get("default_metric", "rmse")
            elif self.task == Task.TIME_SERIES:
                self.metric = self.config.get("default_metric", "rmse")
            else:
                self.metric = "accuracy"
        else:
            self.metric = metric

        print(f"Using metric: {self.metric}")

        self.models_config = self.config.get("models", {})
        self.optimal_model = None
        self.optimal_hyperparameters = {}
        self.metric_value = None

    def build_model(self, candidate_params: dict, y=None): # Added y=None for the ARIMA and SARIMA case
        """
        Build a model instance given:
            candidate_params["model"] -> a string of the model name
            plus any required hyperparameters.

        This method handles classification vs regression automatically.
        """
        from automl.enums import Task

        model_name = candidate_params["model"]
        model_lower = model_name.lower()

        if model_lower == "randomforest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            ModelClass = RandomForestClassifier if self.task == Task.CLASSIFICATION else RandomForestRegressor
            model = ModelClass(
                n_estimators=candidate_params.get("n_estimators"),
                max_depth=candidate_params.get("max_depth"),
                min_samples_split=candidate_params.get("min_samples_split")
            )
        elif model_lower == "xgboost":
            from xgboost import XGBClassifier, XGBRegressor
            ModelClass = XGBClassifier if self.task == Task.CLASSIFICATION else XGBRegressor
            model = ModelClass(
                learning_rate=candidate_params.get("learning_rate"),
                n_estimators=candidate_params.get("n_estimators"),
                max_depth=candidate_params.get("max_depth"),
                gamma=candidate_params.get("gamma"),
            )
        elif model_lower == "logisticregression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                C=candidate_params.get("C")
            )
        elif model_lower == "linearregression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_lower == "ridge":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=candidate_params.get("alpha"))
        elif model_lower == "lasso":
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=candidate_params.get("alpha"))
        elif model_lower == "lightgbm":
            import lightgbm as lgb
            ModelClass = lgb.LGBMClassifier if self.task == Task.CLASSIFICATION else lgb.LGBMRegressor
            model = ModelClass(
                learning_rate=candidate_params.get("learning_rate"),
                n_estimators=candidate_params.get("n_estimators"),
                num_leaves=candidate_params.get("num_leaves"),
                max_depth=candidate_params.get("max_depth"),
                min_child_samples=candidate_params.get("min_child_samples"),
                verbose=-1
            )
        elif model_lower == "arima":
            from statsmodels.tsa.arima.model import ARIMA
            ModelClass = ARIMA if self.task == Task.TIME_SERIES else None
            model = ModelClass(
                endog=y,
                order=(
                    candidate_params.get("p"),
                    candidate_params.get("d"),
                    candidate_params.get("q")
                )
            )
        elif model_lower == "sarimax":
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            ModelClass = SARIMAX if self.task == Task.TIME_SERIES else None
            model = ModelClass(
                endog=y,
                enforce_stationarity=False,
                enforce_invertibility=False,
                order=(

                    candidate_params.get("p"),
                    candidate_params.get("d"),
                    candidate_params.get("q")
                ),
                seasonal_order=(
                    candidate_params.get("P"),
                    candidate_params.get("D"),
                    candidate_params.get("Q"),
                    candidate_params.get("s")
                )
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def evaluate_candidate(self, model_builder, candidate_params, X, y):
        """
        Builds a model (via model_builder), performs cross-validation,
        and returns the average score.
        """
        model = model_builder(candidate_params)

        metric = self.metric
        if metric == "rmse":
            metric = "neg_root_mean_squared_error"

        scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=metric)
        avg_score = scores.mean()

        if self.verbose:
            print(f"Evaluated candidate {candidate_params} => Score: {avg_score:.4f}")
    
        return avg_score
    
    @abstractmethod
    def fit(self, X, y):
        """Run the optimization process and return the fitted model."""
        pass

    def get_optimal_model(self):
        return self.optimal_model

    def get_metric_value(self):
        return self.metric_value