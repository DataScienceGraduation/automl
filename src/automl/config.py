import numpy as np
from automl.enums import Task


CLASSIFICATION_CONFIG = {
    "default_metric": "accuracy",
    "models": {
        "RandomForest": {
            "n_estimators": list(np.arange(100, 300, 10)),
            "max_depth": list(np.arange(5, 30, 1)),
            "min_samples_split": list(np.arange(2, 10, 2))
        },
        "XGBoost": {
            "learning_rate": list(np.arange(0.001, 0.1, 0.001)),
            "n_estimators": list(np.arange(100, 300, 10)),
            "max_depth": list(np.arange(5, 30, 1)),
            "gamma": list(np.arange(0.1, 10, 0.1))
        },
        "LogisticRegression": {
            "C": list(np.arange(0.1, 10, 0.1))
        }
    }
}


REGRESSION_CONFIG = {
    "default_metric": "rmse",
    "models": {
        "RandomForest": {
            "n_estimators": list(np.arange(100, 300, 10)),
            "max_depth": list(np.arange(5, 30, 1)),
            "min_samples_split": list(np.arange(2, 10, 2))
        },
        "XGBoost": {
            "learning_rate": list(np.arange(0.001, 0.1, 0.001)),
            "n_estimators": list(np.arange(100, 300, 10)),
            "max_depth": list(np.arange(5, 30, 1)),
            "gamma": list(np.arange(0.1, 10, 0.1))
        },
        "LinearRegression": {},
        "Ridge": {
            "alpha": list(np.arange(0.1, 10, 0.1))
        },
        "Lasso": {
            "alpha": list(np.arange(0.1, 10, 0.1))
        }
    }
}


def get_config(task: str):
    if task == Task.CLASSIFICATION:
        return CLASSIFICATION_CONFIG
    elif task == Task.REGRESSION:
        return REGRESSION_CONFIG
    else:
        raise ValueError("Unsupported task type")
