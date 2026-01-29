import pytest
import numpy as np
from models.roas import ROASCalculator

def test_roas_calc():
    cpi = 2.0
    days = [30, 60, 90]
    ltv = [2.0, 3.0, 4.0] # 2.0 * 0.7 = 1.4 < 2.0; 3.0 * 0.7 = 2.1 > 2.0
    
    calc = ROASCalculator(cpi, days, ltv)
    roas = calc.calculate_roas()
    
    assert roas[0] < 1.0
    assert roas[1] > 1.0
    
    payback = calc.get_payback_period()
    assert payback == 60

def test_roas_metrics():
    cpi = 2.0
    days = [10, 20]
    ltv = [1.0, 2.0]
    
    calc = ROASCalculator(cpi, days, ltv)
    metrics = calc.get_metrics_at_days([10, 15])
    assert "D10 ROAS" in metrics
    assert "D15 ROAS" in metrics
    assert metrics["D10 ROAS"] == (1.0 * 0.7) / 2.0
