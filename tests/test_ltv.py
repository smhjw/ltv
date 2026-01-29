import pytest
import numpy as np
from models.ltv import LTVModel
from models.retention import RetentionModel

def test_power_law_fit():
    days = [1, 2, 3, 7, 14, 30]
    ltv = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    
    model = LTVModel(model_type='power_law')
    model.fit(days, ltv)
    
    assert model.params is not None
    pred = model.predict([60])
    assert pred[0] > 3.0 # Should grow

def test_retention_based_ltv():
    days = [1, 2, 3, 7, 14, 30]
    retention = [0.5, 0.4, 0.35, 0.25, 0.20, 0.15]
    ltv = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    
    ret_model = RetentionModel()
    ret_model.fit(days, retention)
    
    model = LTVModel(model_type='retention_based')
    model.fit(days, ltv, retention_model=ret_model)
    
    pred = model.predict([60])
    assert pred[0] > 3.0
    
    sens = model.sensitivity_analysis([60])
    assert sens is not None
    assert len(sens) == 2 # 80% and 120% retention scenarios

