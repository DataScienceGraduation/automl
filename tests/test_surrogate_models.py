import numpy as np
import pytest
from automl.optimizers.surrogate_models.gaussian_process_surrogate import GaussianProcessSurrogate
from automl.enums import Task

def test_gaussian_process_surrogate_classification():
    gp = GaussianProcessSurrogate(Task.CLASSIFICATION)
    from sklearn.gaussian_process import GaussianProcessClassifier
    assert isinstance(gp.model, GaussianProcessClassifier)
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    gp.fit(X, y)
    with pytest.raises(NotImplementedError):
        gp.predict(X, return_std=True)
    labels, std = gp.predict(X, return_std=False)
    assert std is None
    assert labels.shape == (2,)

def test_gaussian_process_surrogate_regression():
    gp = GaussianProcessSurrogate(Task.REGRESSION)
    from sklearn.gaussian_process import GaussianProcessRegressor
    assert isinstance(gp.model, GaussianProcessRegressor)
    X = np.array([[0], [1]])
    y = np.array([0.0, 1.0])
    gp.fit(X, y)
    y_mean, y_std = gp.predict(X, return_std=True)
    assert isinstance(y_mean, np.ndarray) and isinstance(y_std, np.ndarray)
    y_mean2, std2 = gp.predict(X, return_std=False)
    assert std2 is None
    assert y_mean2.shape == (2,)
