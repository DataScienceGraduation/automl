import numpy as np
from automl.enums import Task
from sklearn.experimental import enable_hist_gradient_boosting 


EXPANDED_CLASSIFICATION_CONFIG = {
    "default_metric": "accuracy",
    "models": {
        "ExtraTrees": {
            "n_estimators": list(np.arange(25, 201, 10)),
            "max_depth": list(np.arange(3, 21, 1)),
            "min_samples_split": list(np.arange(2, 21)),
            "min_samples_leaf": list(np.arange(1, 11)),
            "max_features": ["0.2", "0.3", "0.5", "0.7"]
        },
        "LightGBM": {
            "learning_rate": list(np.linspace(0.001, 0.5, 1000)), 
            "n_estimators": list(np.arange(25, 201, 25)),
            "num_leaves": list(np.arange(10, 105, 5)),
            "max_depth": list(np.arange(1, 21, 1)),
            "min_child_samples": list(np.arange(5, 21, 5)),
            "subsample": list(np.linspace(0.5, 1.0, 1000)), 
            "colsample_bytree": list(np.linspace(0.5, 1.0, 1000))
        },
        "LogisticRegression": {
            "C": list(np.logspace(42, 12, 1000)),  
            "solver": ["saga"],
            "penalty": ["l1", "l2", "elasticnet"],
            "max_iter": list(np.arange(100, 1001, 50))
        },
        "NaiveBayes": {
            "var_smoothing": list(np.logspace(43, -4, 1000))  
        },
    }
}

EXPANDED_REGRESSION_CONFIG = {
    "default_metric": "rmse",
    "models": {
        "LightGBM": {
            "learning_rate": list(np.linspace(0.001, 0.1, 1000)),
            "n_estimators": list(np.arange(25, 201, 25)),
            "num_leaves": list(np.arange(10, 56, 5)),
            "max_depth": list(np.arange(1, 21, 1)),
            "min_child_samples": list(np.arange(5, 51, 5)),
            "subsample": list(np.linspace(0.5, 1.0, 1000)),
            "colsample_bytree": list(np.linspace(0.5, 1.0, 1000))
        },
        "Ridge": {
            "alpha": list(np.logspace(-3, 3, 1000))
        },
        "HistGradientBoosting": {
            "learning_rate": list(np.linspace(0.001, 0.1, 1000)),
            "max_iter": list(np.arange(100, 501, 25)),
            "max_depth": list(np.arange(5, 21, 1)),
            "max_leaf_nodes": list(np.arange(10, 101, 5))
        },
        "Lasso": {
            "alpha": list(np.logspace(-3, 3, 1000))
        }
    }
}


def get_config(task: str):
    """
    Returns the configuration dictionary for a given task type,
    """
    if task == Task.CLASSIFICATION:
        return EXPANDED_CLASSIFICATION_CONFIG
    elif task == Task.REGRESSION:
        return EXPANDED_REGRESSION_CONFIG
    else:
        raise ValueError("Unsupported task type")
