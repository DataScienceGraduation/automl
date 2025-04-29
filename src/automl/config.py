import numpy as np
from automl.enums import Task


CLASSIFICATION_CONFIG = {
    "default_metric": "accuracy",
    "models": {
        "RandomForest": {
            "n_estimators": list(np.arange(100, 501, 50)),
            "max_depth": list(np.arange(3, 31, 1)),
            "min_samples_split": list(np.arange(2, 11, 1)),
            "min_samples_leaf": list(np.arange(1, 6, 1)),
            "bootstrap": [True, False]
        },
        "XGBoost": {
            "learning_rate": list(np.linspace(0.0001, 0.3, 30)),
            "n_estimators": list(np.arange(50, 501, 50)),
            "max_depth": list(np.arange(3, 16, 1)),
            "gamma": list(np.linspace(0, 10, 21)),
            "subsample": list(np.linspace(0.6, 1.0, 5)),
            "colsample_bytree": list(np.linspace(0.6, 1.0, 5))
        },
        "LightGBM": {
            "learning_rate": list(np.linspace(0.0001, 0.3, 30)),
            "n_estimators": list(np.arange(50, 501, 50)),
            "num_leaves": list(np.arange(31, 256, 25)),
            "max_depth": list(np.arange(-1, 17, 1)),
            "min_child_samples": list(np.arange(5, 51, 5))
        },
        "LogisticRegression": {
            "C": list(np.logspace(-2, 2, 10)),
            "solver": ["lbfgs", "sag", "saga"],
            "penalty": ["l2", "none"]
        }
    }
}


REGRESSION_CONFIG = {
    "default_metric": "rmse",
    "models": {
        "RandomForest": {
            "n_estimators": list(np.arange(100, 501, 50)),
            "max_depth": list(np.arange(3, 31, 1)),
            "min_samples_split": list(np.arange(2, 11, 1)),
            "min_samples_leaf": list(np.arange(1, 6, 1)),
            "bootstrap": [True, False]
        },
        "XGBoost": {
            "learning_rate": list(np.linspace(0.0001, 0.3, 30)),
            "n_estimators": list(np.arange(50, 501, 50)),
            "max_depth": list(np.arange(3, 16, 1)),
            "gamma": list(np.linspace(0, 10, 21)),
            "subsample": list(np.linspace(0.6, 1.0, 5)),
            "colsample_bytree": list(np.linspace(0.6, 1.0, 5))
        },
        "LightGBM": {
            "learning_rate": list(np.linspace(0.0001, 0.3, 30)),
            "n_estimators": list(np.arange(50, 501, 50)),
            "num_leaves": list(np.arange(31, 256, 25)),
            "max_depth": list(np.arange(-1, 17, 1)),
            "min_child_samples": list(np.arange(5, 51, 5))
        },
        "LinearRegression": {},
        "Ridge": {
            "alpha": list(np.logspace(-2, 2, 10))  # 0.01..100
        },
        "Lasso": {
            "alpha": list(np.logspace(-2, 2, 10))  # 0.01..100
        }
    }
}

TIME_SERIES_CONFIG = {
    "default_metric": "rmse",
    "models": {
        "ARIMA": {
            "p": list(np.arange(0, 7, 1)),
            "d": list(np.arange(0, 3, 1)),
            "q": list(np.arange(0, 7, 1)),
        },
        "SARIMAX": {
            "p": list(np.arange(0, 7, 1)),
            "d": list(np.arange(0, 3, 1)),
            "q": list(np.arange(0, 7, 1)),
            "P": list(np.arange(0, 7, 1)),
            "D": list(np.arange(0, 3, 1)),
            "Q": list(np.arange(0, 7, 1)),
            "s": list(np.arange(1, 13, 1)),
        }
    }
}

<<<<<<< HEAD
CLUSTERING_CONFIG = {
    "default_metric": "silhouette",
    "models": {
        "KMeans": {
            "n_clusters": list(np.arange(2, 11, 1)),
            "random_state": [42]
        },
        "DBSCAN": {
            "eps": list(np.linspace(0.1, 1.0, 10)),
            "min_samples": list(np.arange(1, 11, 1))
        },
    }
}
=======
>>>>>>> b997f23f419e69a11232bdab3a6219625062a99e

def get_config(task: str):
    """
    Returns the configuration dictionary for a given task type.
    """
    if task == Task.CLASSIFICATION:
        return CLASSIFICATION_CONFIG
    elif task == Task.REGRESSION:
        return REGRESSION_CONFIG
    elif task == Task.TIME_SERIES:
        return TIME_SERIES_CONFIG
    else:
        raise ValueError("Unsupported task type")