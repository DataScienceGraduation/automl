from sklearn.pipeline import Pipeline
from pandas import DataFrame
from .feature_engineer import FeatureEngineer
from .preprocess import Preprocess
from .preprocess.TimeSeriesPreprocessor import TimeSeriesPreprocessor
import logging

logger = logging.getLogger(__name__)

def createPipeline(df: DataFrame, target_variable: str = None, task: str = None) -> Pipeline:
    logger.info(f"Creating pipeline for task: {task}")
    
    if task == "TimeSeries":
        logger.debug("Using TimeSeriesPreprocessor")
        preprocessor = TimeSeriesPreprocessor(target_column=target_variable)
    else:
        logger.debug("Using standard Preprocess")
        preprocessor = Preprocess(target_variable=target_variable)

    # For clustering, we don't need feature engineering with target variable
    if task == "Clustering":
        logger.debug("Creating clustering pipeline without feature engineering")
        pipeline = Pipeline([
            ('preprocess', preprocessor)
        ])
    else:
        logger.debug("Creating standard pipeline with feature engineering")
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('feature_engineer', FeatureEngineer(target_variable=target_variable)),
        ])

    # For clustering, we don't need to fit with target variable
    if task == "Clustering":
        logger.debug("Fitting pipeline for clustering task")
        pipeline.fit(df)
    else:
        logger.debug("Fitting pipeline with target variable")
        pipeline.fit(df, df[target_variable])

    return pipeline