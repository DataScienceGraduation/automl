import os
import pickle
import time
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from automl.optimizers.bayesian_optimizer import BayesianOptimizer
from automl.enums import Task
from pathlib import Path
from automl.functions import createPipeline
from automl.optimizers.surrogate_models import MetaGaussianSurrogate

BASE_DIR = Path(__file__).resolve().parent.parent

data_folder = BASE_DIR / "src/new_datasets"
save_folder = BASE_DIR / "saved_surrogates"
stats_folder = BASE_DIR / "dataset_statistics"
counter_file = save_folder / "gp_counter.txt"

data_folder = str(data_folder)
save_folder = str(save_folder)
stats_folder = str(stats_folder)
counter_file = str(counter_file)


# Process new datasets
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        dataset_path = os.path.join(data_folder, file)
        dataset_name = os.path.splitext(file)[0]
        print(f"Processing new dataset: {dataset_name}")

        df = pd.read_csv(dataset_path)
        print(df.shape)
        target_variable = df['target']
        num_categorical = len(df.select_dtypes(include=["object", "category"]).columns)

        pipeline = createPipeline(df, target_variable)
        df = pipeline.transform(df)
        X = df.drop(columns=['target'])
        y = df['target'].values

        task = Task.REGRESSION

        # Compute statistics for the new dataset
        num_samples = X.shape[0]
        num_features = X.shape[1]
        num_numerical = num_features - num_categorical
        feature_means = X.select_dtypes(include=[np.number]).mean().mean()
        feature_stds = X.select_dtypes(include=[np.number]).std().mean()
        features_skewness = X.skew().mean()
        features_kurtosis = X.kurtosis().mean()

        if task == Task.REGRESSION:
            target_entropy = np.nan
            imbalance_ratio = np.nan
        else:
            value_counts = pd.Series(y).value_counts(normalize=True)
            target_entropy = stats.entropy(value_counts)
            imbalance_ratio = round(value_counts.max() / value_counts.min(), 2) if len(value_counts) > 1 else 1.0

        # Train optimizer with knowledge transfer
        optimizer = BayesianOptimizer(task=task, surrogate_model=MetaGaussianSurrogate(task, df),time_budget=10800)

        start_time = time.time()
        final_model = optimizer.fit(X, y)
        duration = time.time() - start_time
        print("Done fitting the optimizer on new data")

        # Save new surrogate model
        if os.path.exists(counter_file):
            with open(counter_file, "r") as f:
                try:
                    gp_counter = int(f.read().strip()[-1])     
                except ValueError:
                    gp_counter = 0                
        else:
            gp_counter = 0 


        surrogate_path = os.path.join(save_folder, f"{dataset_name}_transfer_{gp_counter}.pkl")
        with open(surrogate_path, "wb") as f:
            pickle.dump(optimizer.surrogate_model, f)

        gp_counter += 1
        with open(counter_file, "w") as f:
            f.write(str(gp_counter))

        stats = {
            "dataset": dataset_name,
            "task": task.name,
            "num_samples": num_samples,
            "num_features": num_features,
            "num_numerical_features": num_numerical,
            "num_categorical_features": num_categorical,
            "feature_skewness": round(features_skewness, 4),
            "feature_kurtosis": round(features_kurtosis, 4),
            "feature_mean_mean": round(feature_means, 4),
            "feature_std_mean": round(feature_stds, 4),
            "best_score": round(optimizer.best_score, 4),
            "duration_sec": round(duration, 2),
            "target_entropy": round(target_entropy, 4) if not np.isnan(target_entropy) else np.nan,
            "imbalance_ratio": imbalance_ratio,
        }

        stats_df = pd.DataFrame([stats]) 
        stats_path = os.path.join(stats_folder, f"{dataset_name}_transfer_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Statistics saved to {stats_path}")

print("Knowledge transfer complete. New models saved.")
