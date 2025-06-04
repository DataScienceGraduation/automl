import pytest
from automl.optimizers.base_optimizer import BaseOptimizer, MODEL_MAPPING
from automl.enums import Task

class DummyOptimizer(BaseOptimizer):
    def fit(self, X, y):
        pass

def test_map_candidate_model_int_to_str():
    dummy = DummyOptimizer(task=Task.CLASSIFICATION, time_budget=1, config={'models':{}})
    params = {'model': 0}
    mapped = dummy._map_candidate_model(params.copy())
    assert mapped['model'] == MODEL_MAPPING[0]

def test_map_candidate_model_no_change():
    dummy = DummyOptimizer(task=Task.CLASSIFICATION, time_budget=1, config={'models':{}})
    params = {'model': 'LightGBM'}
    mapped = dummy._map_candidate_model(params.copy())
    assert mapped['model'] == 'LightGBM'

def test_build_model_randomforest_classification():
    dummy = DummyOptimizer(task=Task.CLASSIFICATION, time_budget=1, config={'models':{}})
    params = {'model': 'RandomForest', 'n_estimators':2, 'max_depth':1}
    model = dummy.build_model(params.copy())
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)

def test_build_model_randomforest_regression():
    dummy = DummyOptimizer(task=Task.REGRESSION, time_budget=1, config={'models':{}})
    params = {'model': 'RandomForest', 'n_estimators':2, 'max_depth':1}
    model = dummy.build_model(params.copy())
    from sklearn.ensemble import RandomForestRegressor
    assert isinstance(model, RandomForestRegressor)

def test_build_model_unsupported():
    dummy = DummyOptimizer(task=Task.CLASSIFICATION, time_budget=1, config={'models':{}})
    with pytest.raises(ValueError):
        dummy.build_model({'model':'UnknownModel'})

def test_build_model_histgradientboosting_classification():
    dummy = DummyOptimizer(task=Task.CLASSIFICATION, time_budget=1, config={'models':{}})
    params = {
        'model': 'HistGradientBoosting',
        'learning_rate': 0.05,
        'max_iter': 10
    }
    model = dummy.build_model(params.copy())
    from sklearn.ensemble import HistGradientBoostingClassifier
    assert isinstance(model, HistGradientBoostingClassifier)

def test_build_model_lightgbm_classification():
    pytest.importorskip("lightgbm")
    dummy = DummyOptimizer(task=Task.CLASSIFICATION, time_budget=1, config={'models': {}})
    params = {
        'model': 'LightGBM',
        'learning_rate': 0.05,
        'n_estimators': 10
    }
    model = dummy.build_model(params.copy())
    import lightgbm as lgb
    assert isinstance(model, lgb.LGBMClassifier)
