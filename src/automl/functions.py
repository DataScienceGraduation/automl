from sklearn.pipeline import Pipeline
from pandas import DataFrame
from .feature_engineer import FeatureEngineer
from .preprocess import Preprocess


def createPipeline(df: DataFrame, target_variable: str) -> Pipeline:
    print("Creating the pipeline")
    pipeline = Pipeline([
        ('preprocess', Preprocess(target_variable=target_variable)),
        ('feature_engineer', FeatureEngineer(target_variable=target_variable)),
    ])

    pipeline.fit(df, df[target_variable])

    return pipeline
