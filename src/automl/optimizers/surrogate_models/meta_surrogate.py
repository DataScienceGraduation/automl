from .base_surrogate_model import BaseSurrogateModel
from typing import Tuple, Any
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.metrics.pairwise import cosine_similarity
from automl.enums import Task
from pathlib import Path
from automl.optimizers.surrogate_models.gaussian_process_surrogate import GaussianProcessSurrogate

BASE_DIR = Path(__file__).resolve().parents[4]
save_folder = BASE_DIR / "saved_surrogates"
stats_folder = BASE_DIR / "dataset_statistics"

class MetaGaussianSurrogate(BaseSurrogateModel):
    def __init__(self, task: Task, data: pd.DataFrame):
        """
        A unified Gaussian Process surrogate that branches to GPR or GPC:
            - Classification -> GaussianProcessClassifier with Matern kernel.
            - Regression -> GaussianProcessRegressor with RBF kernel.
        This version integrates past surrogate models using weighted averaging.
        """
        self.task = task
        self.data = data
        self.save_folder = str(save_folder)
        self.stats_folder = str(stats_folder)

        # Define the base kernel for the new model
        if self.task == Task.CLASSIFICATION:
            kernel = C(1.0, (1e-3, 1e5)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-3, 1e5))
            self.model = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=20)
        else:
            kernel = C(1.0, (1e-3, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e5))
            self.model = GaussianProcessRegressor(kernel=kernel, alpha=1e-7, n_restarts_optimizer=20)

        # Load past dataset statistics
        past_stats = []
        for file in os.listdir(self.stats_folder):
            if file.endswith("_stats.csv"):
                stats_path = os.path.join(self.stats_folder, file)
                df = pd.read_csv(stats_path)
                past_stats.append(df.iloc[0])

        if not past_stats:
            print("No past dataset statistics found. Proceeding without knowledge transfer.")
            return

        past_stats_df = pd.DataFrame(past_stats)

        # Compute new dataset statistics
        num_samples = data.shape[0]
        num_features = data.shape[1]
        num_numerical = len(data.select_dtypes(include=[np.number]).columns)
        num_categorical = num_features - num_numerical
        feature_means = data.select_dtypes(include=[np.number]).mean().mean()
        feature_stds = data.select_dtypes(include=[np.number]).std().mean()
        features_skewness = data.skew().mean()
        features_kurtosis = data.kurtosis().mean()

        new_stats = np.array([
            num_samples, num_features, num_numerical, num_categorical,
            feature_means, feature_stds, features_skewness, features_kurtosis
        ]).reshape(1, -1)

        # Compute similarity with past datasets
        past_features = past_stats_df[[
            "num_samples", "num_features", "num_numerical_features", "num_categorical_features",
            "feature_mean_mean", "feature_std_mean", "feature_skewness", "feature_kurtosis"
        ]].values

        similarities = cosine_similarity(new_stats, past_features)[0]
        weights = similarities / np.sum(similarities)

        print(f"Dataset similarities: {dict(zip(past_stats_df['dataset'], similarities))}")

        # Load and combine past surrogate models
        combined_model = None
        for i, past_dataset in enumerate(past_stats_df["dataset"]):
            surrogate_path = os.path.join(self.save_folder, f"{past_dataset}_{i}.pkl")
            if os.path.exists(surrogate_path):
                with open(surrogate_path, "rb") as f:
                    past_model = pickle.load(f)
                    if combined_model is None:
                        combined_model = past_model
                        combined_model.model.kernel_ = weights[i] * past_model.model.kernel_
                        combined_model.model.alpha = weights[i] * past_model.model.alpha
                    else:
                        combined_model.model.kernel_ += weights[i] * past_model.model.kernel_
                        combined_model.model.alpha += weights[i] * past_model.model.alpha

        # Assign the combined model
        if combined_model is not None:
            self.combined_surrogate_model = combined_model.model  # Use the combined surrogate model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit either a GPC or GPR depending on the task.
        """
        new_model = GaussianProcessSurrogate(self.task)
        new_model.fit(X, y)
        self.model.kernel_ = 0.5 * self.combined_surrogate_model.kernel_ + 0.5 * new_model.model.kernel_
        self.model.alpha = 0.5 * self.combined_surrogate_model.alpha + 0.5 * new_model.model.alpha


    def predict(self, X: np.ndarray, return_std: bool = False) -> tuple[Any, None] | tuple[Any, Any]:
        """
        - For regression (GPR): returns (mean, std) if return_std=True.
        - For classification (GPC):
            -> predict() returns discrete class labels.
               If return_std=True, raises NotImplementedError.
        """
        if self.task == Task.CLASSIFICATION:
            if return_std:
                raise NotImplementedError(
                    "GaussianProcessClassifier doesn't provide std dev directly. "
                    "Use predict_proba or a custom uncertainty measure if needed."
                )
            labels = self.model.predict(X)
            return labels, None
        else:
            if return_std:
                y_mean, y_std = self.model.predict(X, return_std=True)
                return y_mean, y_std
            else:
                y_mean = self.model.predict(X, return_std=False)
                return y_mean, None
