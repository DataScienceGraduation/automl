from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score
import numpy as np
from automl import Task
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_MAPPING = {
    0: "RandomForest",
    1: "XGBoost",
    2: "LightGBM",
    3: "LogisticRegression",
    4: "Ridge",               
    5: "HistGradientBoosting",
    6: "ExtraTrees",
    7: "NaiveBayes",
    8: "LinearRegression",
    9: "Lasso"
}



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
            else:
                self.metric = "accuracy"
        else:
            self.metric = metric

        print(f"Using metric: {self.metric}")

        self.models_config = self.config.get("models", {})
        self.optimal_model = None
        self.optimal_hyperparameters = {}
        self.metric_value = None

    def _map_candidate_model(self, candidate_params: dict) -> dict:
        """
        Convert candidate_params["model"] from a numeric ID to a string name,
        if necessary.
        """
        model_value = candidate_params.get("model")
        if isinstance(model_value, (int, np.integer)):
            # Look up the corresponding model name using our mapping.
            candidate_params["model"] = MODEL_MAPPING[int(model_value)]
            if self.verbose:
                logger.info(f"Mapping numeric model {model_value} to "
                            f"{candidate_params['model']}.")
        return candidate_params

    def build_model(self, candidate_params: dict):
        """
        Build a model instance given candidate_params.
        The candidate_params dictionary should contain a "model" key
        that names the model (string) along with its hyperparameters.
        """
        # Ensure candidate_params["model"] is a string:
        candidate_params = self._map_candidate_model(candidate_params)
        model_name = candidate_params["model"]
        model_lower = model_name.lower()
        for key, value in candidate_params.items():
            if isinstance(value, float) and value.is_integer():
                candidate_params[key] = int(value)
            elif isinstance(value, str) and value.isdigit():
                candidate_params[key] = float(value)

        if model_lower == "randomforest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            ModelClass = (RandomForestClassifier if self.task == Task.CLASSIFICATION 
                          else RandomForestRegressor)
            model = ModelClass(
                n_estimators=candidate_params.get("n_estimators", 100),
                max_depth=candidate_params.get("max_depth", None),
                min_samples_split=candidate_params.get("min_samples_split", 2),
                n_jobs=-1  # enable parallelism
            )

        elif model_lower == "xgboost":
            from xgboost import XGBClassifier, XGBRegressor
            ModelClass = (XGBClassifier if self.task == Task.CLASSIFICATION 
                          else XGBRegressor)
            model = ModelClass(
                learning_rate=candidate_params.get("learning_rate", 0.1),
                n_estimators=candidate_params.get("n_estimators", 100),
                max_depth=candidate_params.get("max_depth", None),
                gamma=candidate_params.get("gamma", 0),
                n_jobs=-1 # parallelism for XGBoost
            )

        elif model_lower == "lightgbm":
            import lightgbm as lgb
            ModelClass = (lgb.LGBMClassifier if self.task == Task.CLASSIFICATION 
                        else lgb.LGBMRegressor)
            model = ModelClass(
                learning_rate=candidate_params.get("learning_rate", 0.1),
                n_estimators=candidate_params.get("n_estimators", 100),
                num_leaves=candidate_params.get("num_leaves", 31),
                max_depth=candidate_params.get("max_depth", -1),
                min_child_samples=candidate_params.get("min_child_samples", 20),
                subsample=candidate_params.get("subsample", 1.0),
                colsample_bytree=candidate_params.get("colsample_bytree", 1.0),
                reg_alpha=candidate_params.get("reg_alpha", 0.0),
                reg_lambda=candidate_params.get("reg_lambda", 0.0),
                n_jobs=-1,
                verbose=-1
            )

        elif model_lower == "logisticregression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                solver=candidate_params.get("solver", "saga"),
                C=candidate_params.get("C", 1.0),
                penalty=candidate_params.get("penalty", "l2"),
                max_iter=candidate_params.get("max_iter", 100),
                n_jobs=-1
            )

        elif model_lower == "ridge":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=candidate_params.get("alpha", 1.0), max_iter=candidate_params.get("max_iter", 1000))
        
        elif model_lower == "lasso":
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=candidate_params.get("alpha", 1.0), max_iter=candidate_params.get("max_iter", 1000))

        elif model_lower == "linearregression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(n_jobs=-1)

        elif model_lower == "histgradientboosting":
            from sklearn.experimental import enable_hist_gradient_boosting  # noqa
            if self.task == Task.CLASSIFICATION:
                from sklearn.ensemble import HistGradientBoostingClassifier as ModelClass
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor as ModelClass

            model = ModelClass(
                learning_rate=candidate_params.get("learning_rate", 0.1),
                max_iter=candidate_params.get("max_iter", 100),
                max_depth=candidate_params.get("max_depth", None),
                max_leaf_nodes=candidate_params.get("max_leaf_nodes", None),
                min_samples_leaf=candidate_params.get("min_samples_leaf", 20),
                l2_regularization=candidate_params.get("l2_regularization", 0.0),
                early_stopping=True,
                n_iter_no_change=5
            )

        elif model_lower == "extratrees":
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
            ModelClass = (ExtraTreesClassifier if self.task == Task.CLASSIFICATION 
                          else ExtraTreesRegressor)
            model = ModelClass(
                n_estimators=candidate_params.get("n_estimators", 50),
                max_depth=candidate_params.get("max_depth", None),
                min_samples_split=candidate_params.get("min_samples_split", 2),
                n_jobs=-1
            )

        elif model_lower == "naivebayes":
            if self.task == Task.CLASSIFICATION:
                from sklearn.naive_bayes import GaussianNB
                model = GaussianNB(
                    var_smoothing=candidate_params.get("var_smoothing", 1e-9)
                )
            else:
                raise ValueError("NaiveBayes is not typical for regression.")
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