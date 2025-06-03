import pandas as pd
import numpy as np
import pytest
from automl.preprocess import Preprocess, DropSingleValueColumns, DropHighMissing, DropHighCardinality, DropNullRows, RemoveDuplicates, CustomLabelEncoder, RemoveOutliers

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'const': [1, 1, 1, 1],
        'values': [1, 2, 3, 4],
        'high_missing': [1, None, None, None],
        'high_card': ['a', 'b', 'c', 'd'],
        'cat': ['x', 'y', 'x', 'z'],
        'target': [0, 1, 0, 1]
    })

def test_drop_single_value_columns(sample_df):
    transformer = DropSingleValueColumns()
    transformer.fit(sample_df)
    transformed = transformer.transform(sample_df)
    assert 'const' not in transformed.columns

def test_drop_high_missing(sample_df):
    transformer = DropHighMissing(threshold=0.5)
    transformer.fit(sample_df)
    transformed = transformer.transform(sample_df)
    assert 'high_missing' not in transformed.columns

def test_drop_high_cardinality(sample_df):
    transformer = DropHighCardinality(cardinality_threshold=0.8)
    transformer.fit(sample_df)
    transformed = transformer.transform(sample_df)
    assert 'high_card' not in transformed.columns

def test_drop_null_rows(sample_df):
    transformer = DropNullRows()
    transformer.fit(sample_df)
    transformed = transformer.transform(sample_df)
    assert transformed.isnull().sum().sum() == 0

def test_remove_duplicates():
    df = pd.DataFrame({'a': [1,1,2], 'b': [3,3,4]})
    transformer = RemoveDuplicates()
    transformer.fit(df)
    transformed = transformer.transform(df)
    assert len(transformed) == 2

def test_custom_label_encoder(sample_df):
    transformer = CustomLabelEncoder()
    transformer.fit(sample_df)
    transformed = transformer.transform(sample_df)
    assert transformed['cat'].dtype == np.int64 or np.int32

def test_remove_outliers_removes_expected_fraction():
    df = pd.DataFrame({'a': [1,2,3,4], 'b': [4,5,6,7]})
    transformer = RemoveOutliers(contamination=0.5)
    transformer.fit(df)
    transformed = transformer.transform(df)
    assert len(transformed) == 2

def test_preprocess_pipeline(sample_df):
    pipeline = Preprocess(target_variable='target')
    pipeline.fit(sample_df)
    transformed = pipeline.transform(sample_df)
    assert 'const' not in transformed.columns
    assert 'high_missing' not in transformed.columns
    assert 'high_card' not in transformed.columns
    assert transformed.isnull().sum().sum() == 0
