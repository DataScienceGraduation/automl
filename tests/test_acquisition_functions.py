import numpy as np
import pytest
from automl.optimizers.acquisition_functions.expected_improvement import ExpectedImprovement
from automl.optimizers.acquisition_functions.base_acquisition_function import BaseAcquistionFunction


class DummySurrogate:
    def predict(self, X, return_std=True):
        return np.array([1.0, 2.0]), np.array([0.5, 0.5])


def test_expected_improvement_array_input():
    ei = ExpectedImprovement()
    X = np.array([[1], [2]])
    surrogate = DummySurrogate()
    result = ei.evaluate(X, surrogate, best_observed=0.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


def test_expected_improvement_list_input():
    ei = ExpectedImprovement()
    X = [{'a': 1}, {'a': 2}]
    surrogate = DummySurrogate()
    result = ei.evaluate(X, surrogate, best_observed=1.5)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


def test_expected_improvement_invalid_input():
    ei = ExpectedImprovement()
    with pytest.raises(Exception):
        ei.evaluate('invalid', DummySurrogate(), best_observed=0.0)
