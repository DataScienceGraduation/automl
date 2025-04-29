from sklearn.pipeline import Pipeline
from pandas import DataFrame
from .feature_engineer import FeatureEngineer
from .preprocess import Preprocess
from .preprocess.TimeSeriesPreprocessor import TimeSeriesPreprocessor


def createPipeline(df: DataFrame, target_variable: str, task=None) -> Pipeline:
    print("Creating the pipeline")
    if task == "time series":
        preprocessor = TimeSeriesPreprocessor(target_column=target_variable)
    else:
        preprocessor = Preprocess(target_variable=target_variable)

    pipeline = Pipeline([
        ('preprocess', preprocessor(target_variable=target_variable)),
        ('feature_engineer', FeatureEngineer(target_variable=target_variable)),
    ])

    pipeline.fit(df, df[target_variable])

    return pipeline