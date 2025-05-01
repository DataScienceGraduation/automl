from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score
import numpy as np
from automl import Task
import time
import logging
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseOptimizer(ABC):
    def __init__(self, task, time_budget, metric=None, verbose=False, cv_folds=10, config=None):
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
            elif self.task == Task.CLUSTERING:
                self.metric = self.config.get("default_metric", "silhouette")
            else:
                self.metric = "accuracy"
        else:
            self.metric = metric

        print(f"Using metric: {self.metric}")

        self.models_config = self.config.get("models", {})
        self.optimal_model = None
        self.optimal_hyperparameters = {}
        self.metric_value = None

    def build_model(self, candidate_params: dict, y=None):
        from automl.enums import Task
        model_name = candidate_params["model"].lower()

        if model_name == "randomforest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            ModelClass = RandomForestClassifier if self.task == Task.CLASSIFICATION else RandomForestRegressor
            model = ModelClass(
                n_estimators=candidate_params.get("n_estimators"),
                max_depth=candidate_params.get("max_depth"),
                min_samples_split=candidate_params.get("min_samples_split")
            )
        elif model_name == "xgboost":
            from xgboost import XGBClassifier, XGBRegressor
            ModelClass = XGBClassifier if self.task == Task.CLASSIFICATION else XGBRegressor
            model = ModelClass(
                learning_rate=candidate_params.get("learning_rate"),
                n_estimators=candidate_params.get("n_estimators"),
                max_depth=candidate_params.get("max_depth"),
                gamma=candidate_params.get("gamma"),
            )
        elif model_name == "logisticregression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=candidate_params.get("C"))
        elif model_name == "linearregression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_name == "ridge":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=candidate_params.get("alpha"))
        elif model_name == "lasso":
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=candidate_params.get("alpha"))
        elif model_name == "lightgbm":
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
        elif model_name == "arima":
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(
                endog=y,
                order=(
                    candidate_params.get("p"),
                    candidate_params.get("d"),
                    candidate_params.get("q")
                ),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        elif model_name == "sarimax":
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(
                endog=y,
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
                ),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        elif model_name == "kmeans":
            from sklearn.cluster import KMeans
            model = KMeans(
                n_clusters=candidate_params.get("n_clusters"),
                random_state=candidate_params.get("random_state")
            )
        elif model_name == "dbscan":
            from sklearn.cluster import DBSCAN
            model = DBSCAN(
                eps=candidate_params.get("eps"),
                min_samples=candidate_params.get("min_samples")
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def evaluate_candidate(self, model_builder, candidate_params, X, y=None):
        """
        Builds and evaluates a model. Handles:
        - Supervised: cross-validation
        - Clustering: silhouette score
        - Time Series: negative RMSE on forecast
        """
        if self.task == Task.CLUSTERING:
            from sklearn.metrics import silhouette_score
            model = model_builder(candidate_params)
            model.fit(X)
            labels = model.labels_

            if -1 in labels:
                mask = labels != -1
                if np.sum(mask) > 0:
                    score = silhouette_score(X[mask], labels[mask])
                else:
                    score = -1
            else:
                score = silhouette_score(X, labels)

            if self.verbose:
                print(f"Evaluated clustering {candidate_params} => Score: {score:.4f}")
            return score

        elif self.task == Task.TIME_SERIES:
            n_splits = 2 
            tscv = TimeSeriesSplit(n_splits=n_splits)
            rmse_scores = []

            for train_index, test_index in tscv.split(y):
                train, test = y[train_index], y[test_index]

                model = self.build_model(candidate_params, y=train) 
                model_fit = model.fit()

               
                forecast_horizon = len(test) 
                y_pred = model_fit.forecast(steps=forecast_horizon)
                rmse = np.sqrt(np.mean((test - y_pred) ** 2))
                rmse_scores.append(rmse)
                if self.verbose:
                    print(f"Evaluated time series {candidate_params} => RMSE for fold: {rmse:.4f}")
            avg_rmse = np.mean(rmse_scores)
            score = -avg_rmse  
            if self.verbose:
                print(f"Average RMSE across {n_splits} folds: {avg_rmse:.4f}")

            return score

        else:
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