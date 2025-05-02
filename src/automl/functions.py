from sklearn.pipeline import Pipeline
from pandas import DataFrame
from .feature_engineer import FeatureEngineer
from .preprocess import Preprocess
from .preprocess.TimeSeriesPreprocessor import TimeSeriesPreprocessor


def createPipeline(df: DataFrame, target_variable: str = None, task=None) -> Pipeline:
    print("Creating the pipeline")
    if task == "TimeSeries":
        print("10")
        preprocessor = TimeSeriesPreprocessor(target_column=target_variable)
        print("11")
    else:
        preprocessor = Preprocess(target_variable=target_variable)
        print("12")

    # For clustering, we don't need feature engineering with target variable
    if task == "Clustering":
        print("13")
        pipeline = Pipeline([
            ('preprocess', preprocessor)
        ])
        print("14")
    else:
        print("15")
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('feature_engineer', FeatureEngineer(target_variable=target_variable)),
        ])

    # For clustering, we don't need to fit with target variable
    if task == "Clustering":
        print("16")
        pipeline.fit(df)
        print("17")
    else:
        print("18")
        pipeline.fit(df, df[target_variable])
        print("19")

    return pipeline