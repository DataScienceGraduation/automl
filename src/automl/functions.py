from sklearn.pipeline import Pipeline
from pandas import DataFrame
from .feature_engineer import FeatureEngineer
from .preprocess import Preprocess
from .preprocess.TimeSeriesPreprocessor import TimeSeriesPreprocessor


def createPipeline(df: DataFrame, target_variable: str = None, task=None) -> Pipeline:
    print("Creating the pipeline")
    print(task)
    print(task)
    print(task)
    print(task)
    if task == "TimeSeries":
        preprocessor = TimeSeriesPreprocessor(target_column=target_variable)
    else:
        preprocessor = Preprocess(target_variable=target_variable)

    # For clustering, we don't need feature engineering with target variable
    if task == "clustering":
        pipeline = Pipeline([
            ('preprocess', preprocessor)
        ])
    else:
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('feature_engineer', FeatureEngineer(target_variable=target_variable)),
        ])

    # For clustering, we don't need to fit with target variable
    if task == "clustering":
        pipeline.fit(df)
    else:
        pipeline.fit(df, df[target_variable])

    return pipeline