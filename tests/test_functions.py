import pandas as pd
from automl.functions import createPipeline
from sklearn.pipeline import Pipeline

def test_create_pipeline_returns_pipeline():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": ["x", "y", "z"],
        "target": [0, 1, 0]
    })
    pipeline = createPipeline(df, target_variable="target")
    assert isinstance(pipeline, Pipeline)
    transformed = pipeline.transform(df)
    assert "target" in transformed.columns
