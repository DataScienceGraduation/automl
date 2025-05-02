from sklearn.pipeline import Pipeline
from pandas import DataFrame
from .feature_engineer import FeatureEngineer
from .preprocess import Preprocess
from .preprocess.TimeSeriesPreprocessor import TimeSeriesPreprocessor


def createPipeline(df: DataFrame, target_variable: str = None, task=None) -> Pipeline:
    print("Creating the pipeline")
    if task == "TimeSeries":
        preprocessor = TimeSeriesPreprocessor(target_column=target_variable)
    else:
        preprocessor = Preprocess(target_variable=target_variable)

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('feature_engineer', FeatureEngineer(target_variable=target_variable)),
    ])

    if target_variable is not None:
        pipeline.fit(df, df[target_variable])
    else:
        pipeline.fit(df)

    return pipeline