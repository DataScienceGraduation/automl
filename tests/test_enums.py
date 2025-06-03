import pytest
from automl.enums import Task, Metric

def test_task_parse_valid():
    assert Task.parse("Regression") == Task.REGRESSION
    assert Task.parse("Classification") == Task.CLASSIFICATION
    assert Task.parse("Clustering") == Task.CLUSTERING
    assert Task.parse("Time Series") == Task.TIME_SERIES

def test_task_parse_invalid():
    with pytest.raises(KeyError):
        Task.parse("invalid")

def test_metric_enum():
    assert isinstance(Metric.RMSE, Metric)
    assert Metric.ACCURACY.name == "ACCURACY"
    assert Metric.ACCURACY.value == 3
