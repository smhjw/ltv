import pytest
import numpy as np
from models.retention import RetentionModel

def test_weibull_fit():
    days = [1, 2, 3, 7, 14, 30]
    retention = [0.5, 0.4, 0.35, 0.25, 0.20, 0.15]
    
    model = RetentionModel(model_type='weibull')
    model.fit(days, retention)
    
    assert model.params is not None
    metrics = model.get_metrics()
    assert 'R2' in metrics
    assert 'MAPE' in metrics
    assert metrics['R2'] > 0.8 # Should fit reasonably well

def test_lognormal_fit():
    days = [1, 2, 3, 7, 14, 30]
    retention = [0.5, 0.4, 0.35, 0.25, 0.20, 0.15]
    
    model = RetentionModel(model_type='lognormal')
    model.fit(days, retention)
    
    assert model.params is not None
    pred = model.predict([60])
    assert 0 < pred[0] < 0.15 # Should be lower than day 30

def test_interval_prediction():
    days = [1, 2, 3, 7, 14, 30]
    retention = [0.5, 0.4, 0.35, 0.25, 0.20, 0.15]
    model = RetentionModel(model_type='weibull')
    model.fit(days, retention)
    
    mean, lower, upper = model.predict_with_interval([60])
    assert lower[0] <= mean[0] <= upper[0]
