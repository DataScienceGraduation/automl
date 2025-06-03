import pandas as pd
import numpy as np
from automl.feature_engineer import RemoveHighlyCorrelated, CustomStandardScaler

def test_remove_highly_correlated():
    df = pd.DataFrame({
        'x': [1,2,3,4,5],
        'y': [2,4,6,8,10],
        'target': [1,0,1,0,1]
    })
    transformer = RemoveHighlyCorrelated(target_variable='target', correlation_threshold=0.9)
    transformer.fit(df)
    transformed = transformer.transform(df)
    assert len([col for col in ['x','y'] if col in transformed.columns]) == 1

def test_custom_standard_scaler():
    df = pd.DataFrame({
        'a': [1,2,3,4],
        'b': [2,4,6,8],
        'target': [0,1,0,1]
    })
    transformer = CustomStandardScaler(target_variable='target')
    transformer.fit(df)
    transformed = transformer.transform(df)
    assert abs(transformed['a'].mean()) < 1e-6
    assert abs(transformed['a'].std(ddof=0) - 1) < 1e-6
