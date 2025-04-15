import openml
import os
import pandas as pd

# Define the benchmark suite ID
suite_id = 455  # OpenML-CC18

# Create a directory to store the datasets
output_dir = "new_datasets"
os.makedirs(output_dir, exist_ok=True)

# Fetch the benchmark suite
suite = openml.study.get_suite(suite_id)

# Iterate over each task in the suite
for task_id in suite.tasks:
    try:
        # Fetch the task
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        # Retrieve the data
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        # Combine features and target into a single DataFrame
        df = pd.concat([X, y.rename(dataset.default_target_attribute)], axis=1)

        # rename the columns to avoid spaces
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('-', '_')
        df.columns = df.columns.str.replace('(', '')
        df.columns = df.columns.str.replace(')', '')
        df.columns = df.columns.str.replace('.', '_')
        df.columns = df.columns.str.replace('/', '_')

        # rename the target variable
        target_variable = dataset.default_target_attribute
        df.rename(columns={target_variable: 'target'}, inplace=True)
        # check if dataset size is greater than 30mb
        if df.memory_usage(deep=True).sum() > 30 * 1024 * 1024:
            print(f"Skipping dataset {dataset.name} due to size.")
            continue

        # Define the output CSV file path
        csv_filename = f"{dataset.dataset_id}_{dataset.name}.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        # Save the DataFrame to CSV
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_filename}")

    except Exception as e:
        print(f"Failed to process task {task_id}: {e}")
