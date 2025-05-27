import pytest
from automl.config import get_config, EXPANDED_CLASSIFICATION_CONFIG, EXPANDED_REGRESSION_CONFIG
from automl.enums import Task

def test_get_config_classification():
    cfg = get_config(Task.CLASSIFICATION)
    assert cfg is EXPANDED_CLASSIFICATION_CONFIG

def test_get_config_regression():
    cfg = get_config(Task.REGRESSION)
    assert cfg is EXPANDED_REGRESSION_CONFIG

def test_get_config_invalid():
    with pytest.raises(ValueError):
        get_config("invalid")
