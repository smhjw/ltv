import numpy as np
import pandas as pd

class ROASCalculator:
    def __init__(self, cpi, ltv_curve_days, ltv_curve_values):
        """
        Initialize ROAS calculator.
        :param cpi: Cost Per Install
        :param ltv_curve_days: Days corresponding to LTV values
        :param ltv_curve_values: Cumulative LTV values
        """
        self.cpi = cpi
        self.days = np.array(ltv_curve_days)
        self.ltv_values = np.array(ltv_curve_values)
        self.roas_values = None
    
    def calculate_roas(self, revenue_share=0.7):
        """
        Calculate Cumulative ROAS curve.
        ROAS = (LTV * share) / CPI
        """
        self.roas_values = (self.ltv_values * revenue_share) / self.cpi
        return self.roas_values
    
    def get_payback_period(self, target_roas=1.0):
        """
        Find the day where ROAS >= target_roas.
        Returns the day, or None if not reached within prediction range.
        """
        if self.roas_values is None:
            self.calculate_roas()
            
        # Check if reached
        reached_indices = np.where(self.roas_values >= target_roas)[0]
        if len(reached_indices) > 0:
            return self.days[reached_indices[0]]
        return None
    
    def get_metrics_at_days(self, days=[90, 180]):
        """
        Get ROAS at specific days.
        """
        if self.roas_values is None:
            self.calculate_roas()
            
        metrics = {}
        for day in days:
            # Interpolate if day is not exact
            if day in self.days:
                idx = np.where(self.days == day)[0][0]
                metrics[f"D{day} ROAS"] = self.roas_values[idx]
            else:
                metrics[f"D{day} ROAS"] = np.interp(day, self.days, self.roas_values)
        return metrics
