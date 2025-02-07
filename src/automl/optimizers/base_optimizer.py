from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score
from automl import Task


class BaseOptimizer(ABC):
    def __init__(self, task, time_budget, metric=None, verbose=False, cv_folds=10, config=None):
        """
        Parameters:
            task: str - e.g., "classification" or "regression"
            time_budget: int - time allowed for optimization (in seconds)
            metric: str - scoring function (if None, defaults to value in config based on task)
            verbose: bool - whether to print detailed info (default False)
            cv_folds: int - number of cross-validation folds (default 10)
            config: dict - configuration dictionary (e.g., imported from automl.config)
        """
        self.task = task
        self.time_budget = time_budget
        self.verbose = verbose
        self.cv_folds = cv_folds
        self.config = config or {}

        # Set default metric based on task if not provided.
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

        # Expect configuration to define available models and their hyperparameter ranges.
        self.models_config = self.config.get("models", {})
        self.optimal_model = None
        self.optimal_hyperparameters = {}
        self.metric_value = None

    def evaluate_candidate(self, model_builder, candidate_params, X, y):
        """
        Unified evaluation: Builds a model using the given candidate parameters,
        performs cross-validation, and returns the average score.
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
