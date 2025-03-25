import os
import pickle
import time
import pandas as pd
import numpy as np
from scipy import stats
from automl.optimizers.bayesian_optimizer import BayesianOptimizer
from automl.enums import Task
from automl.functions import createPipeline

if __name__ == '__main__':
    data_folder = "/Users/macbookpro/Desktop/University_Projects/Graduation Project/automl/src/datasets"
    save_folder = "/Users/macbookpro/Desktop/University_Projects/Graduation Project/automl/saved_surrogates"
    stats_folder = "/Users/macbookpro/Desktop/University_Projects/Graduation Project/automl/dataset_statistics"
    
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)
    counter_file = os.path.join(save_folder, "gp_counter.txt")

    # Load the counter if it exists, otherwise start from 0
    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            gp_id = int(f.read().strip())
    else:
        gp_id = 0  # Start from 0 if the counter file does not exist

    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            dataset_path = os.path.join(data_folder, file)
            dataset_name = os.path.splitext(file)[0]
            print(f"Processing dataset: {dataset_name}")

            df = pd.read_csv(dataset_path)
            print(df.shape)
            target_variable = df.columns[-1]
            num_categorical = len(df.select_dtypes(include=["object", "category"]).columns)

            pipeline = createPipeline(df, target_variable)
            df = pipeline.transform(df)
            X = df.drop(columns=target_variable)
            y = df[target_variable].values

            task = Task.REGRESSION

            optimizer = BayesianOptimizer(task=task, time_budget=10)

            start_time = time.time()
            final_model = optimizer.fit(X, y)
            duration = time.time() - start_time
            print("Done fitting the optimizer on data")

            current_gp_id = gp_id  # Assign current ID before incrementing
            gp_id += 1  # Increment for the next model

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

            # Save surrogate model with unique counter
            surrogate_path = os.path.join(save_folder, f"{dataset_name}_{current_gp_id}.pkl")
            with open(surrogate_path, "wb") as f:
                pickle.dump(optimizer.surrogate_model, f)

            # Save statistics for the current dataset
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

            stats_df = pd.DataFrame([stats])  # Wrap in a list to create a DataFrame
            stats_path = os.path.join(stats_folder, f"{dataset_name}_stats.csv")
            stats_df.to_csv(stats_path, index=False)
            print(f"Statistics saved to {stats_path}")

    # Save updated counter back to file
    with open(counter_file, "w") as f:
        f.write(str(gp_id))

    print("All datasets processed. Stats saved.")
