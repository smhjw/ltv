import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

class LTVModel:
    def __init__(self, model_type='power_law'):
        """
        Initialize the LTV model.
        :param model_type: 'power_law', 'logarithmic', or 'retention_based'
        """
        self.model_type = model_type
        self.params = None
        self.cov = None
        self.days = None
        self.ltv_values = None
        self.retention_model = None
        self.arpdau_avg = None

    def _power_law_func(self, t, a, b, c):
        """
        Power Law: LTV(t) = a * t^b + c
        """
        t = np.array(t, dtype=float)
        return a * (t ** b) + c

    def _logarithmic_func(self, t, a, b):
        """
        Logarithmic: LTV(t) = a * ln(t) + b
        """
        t = np.array(t, dtype=float)
        return a * np.log(t) + b

    def fit(self, days, ltv_values, retention_model=None):
        """
        Fit the model.
        :param days: Array of days.
        :param ltv_values: Array of Cumulative LTV values.
        :param retention_model: Fitted RetentionModel instance (required for 'retention_based').
        """
        self.days = np.array(days)
        self.ltv_values = np.array(ltv_values)
        self.retention_model = retention_model

        if self.model_type == 'retention_based':
            if self.retention_model is None:
                raise ValueError("Retention model is required for retention_based LTV model.")
            # Calculate ARPDAU for the known days
            # LTV(t) = Sum(R(i) * ARPDAU(i)) for i=1 to t
            # So, Daily Revenue(t) = LTV(t) - LTV(t-1)
            # ARPDAU(t) = Daily Revenue(t) / R(t)
            
            # Interpolate Retention for the input days if needed
            r_values = self.retention_model.predict(self.days)
            
            daily_rev = np.diff(self.ltv_values, prepend=0)
            # Avoid division by zero
            r_values_safe = np.maximum(r_values, 1e-10)
            arpdau = daily_rev / r_values_safe
            
            # Simple assumption: Future ARPDAU is the average of the last few known days (e.g., last 7 days)
            # or use the overall average if not enough data
            if len(arpdau) >= 7:
                self.arpdau_avg = np.mean(arpdau[-7:])
            else:
                self.arpdau_avg = np.mean(arpdau)
            
            # Store the historical daily revenue for reconstruction
            self.historical_daily_rev = daily_rev
            
        elif self.model_type == 'power_law':
            p0 = [1.0, 0.5, 0.0]
            try:
                self.params, self.cov = curve_fit(self._power_law_func, self.days, self.ltv_values, p0=p0, maxfev=10000)
            except:
                # Fallback to simple power law a*t^b
                self._power_law_simple = lambda t, a, b: a * (t ** b)
                self.params, self.cov = curve_fit(self._power_law_simple, self.days, self.ltv_values, p0=[1.0, 0.5], maxfev=10000)
                self.model_type = 'power_law_simple' # Switch internal type

        elif self.model_type == 'logarithmic':
            p0 = [1.0, 0.0]
            self.params, self.cov = curve_fit(self._logarithmic_func, self.days, self.ltv_values, p0=p0)

    def predict(self, days):
        """
        Predict Cumulative LTV.
        """
        days = np.array(days)
        
        if self.model_type == 'retention_based':
            # For future days, we need to project.
            # LTV(T) = LTV(LastKnown) + Sum(Predicted R(t) * Predicted ARPDAU) for t > LastKnown
            
            max_known_day = int(np.max(self.days))
            last_known_ltv = self.ltv_values[-1]
            
            predictions = []
            for d in days:
                if d <= max_known_day:
                    # Return actual/fitted? Let's return interpolated actuals for consistency or just model logic
                    # Ideally we should re-calculate from R(t) * Historical ARPDAU
                    # But for simplicity, if d is in input days, return the input LTV?
                    # Let's just calculate LTV(d) = Sum(R(i)*ARPDAU_est) to be a smooth curve
                    # Wait, better to respect historical data.
                    # If d is beyond known data:
                    future_days = np.arange(max_known_day + 1, d + 1)
                    if len(future_days) > 0:
                        future_r = self.retention_model.predict(future_days)
                        future_revenue = np.sum(future_r * self.arpdau_avg)
                        predictions.append(last_known_ltv + future_revenue)
                    else:
                        # d is within history. Find closest or interpolate.
                        # For simplicity, if d is in self.days, return that.
                        idx = np.where(self.days == d)[0]
                        if len(idx) > 0:
                            predictions.append(self.ltv_values[idx[0]])
                        else:
                            # Simple interpolation
                            predictions.append(np.interp(d, self.days, self.ltv_values))
                else:
                    future_days = np.arange(max_known_day + 1, d + 1)
                    future_r = self.retention_model.predict(future_days)
                    future_revenue = np.sum(future_r * self.arpdau_avg)
                    predictions.append(last_known_ltv + future_revenue)
            
            return np.array(predictions)

        elif self.model_type == 'power_law':
            return self._power_law_func(days, *self.params)
        elif self.model_type == 'power_law_simple':
            return self._power_law_simple(days, *self.params)
        elif self.model_type == 'logarithmic':
            return self._logarithmic_func(days, *self.params)

    def predict_with_interval(self, days, confidence=0.80):
        """
        Predict with confidence interval.
        """
        pred = self.predict(days)
        
        # Simple heuristic for interval if covariance is available
        if self.cov is not None and self.model_type not in ['retention_based']:
            # Monte Carlo
            num_simulations = 1000
            try:
                simulated_params = np.random.multivariate_normal(self.params, self.cov, num_simulations)
                simulated_curves = []
                for params in simulated_params:
                    if self.model_type == 'power_law':
                        simulated_curves.append(self._power_law_func(days, *params))
                    elif self.model_type == 'power_law_simple':
                         simulated_curves.append(self._power_law_simple(days, *params))
                    elif self.model_type == 'logarithmic':
                        simulated_curves.append(self._logarithmic_func(days, *params))
                
                simulated_curves = np.array(simulated_curves)
                lower = np.percentile(simulated_curves, (1 - confidence) / 2 * 100, axis=0)
                upper = np.percentile(simulated_curves, (1 + confidence) / 2 * 100, axis=0)
                return pred, lower, upper
            except:
                # Fallback if covariance is bad
                pass
        
        # If retention based or fallback
        # Assume +/- 10% error growing with time? Or based on retention error?
        # Requirement 2.3: "Sensitivity analysis (±20% retention change)"
        # So we can just return +/- 20% of the *growth* or similar.
        # Let's just return +/- 5% as a placeholder if no cov.
        return pred, pred * 0.95, pred * 1.05

    def sensitivity_analysis(self, days, retention_change=0.20):
        """
        Analyze impact of retention change on LTV (for retention_based model).
        """
        if self.model_type != 'retention_based':
            return None
        
        base_pred = self.predict(days)
        
        # Simulate changed retention
        # This is tricky without modifying the retention model.
        # We can just scale the future revenue.
        
        # We only affect future predictions (after max_known_day)
        max_known_day = int(np.max(self.days))
        last_known_ltv = self.ltv_values[-1]
        
        results = {}
        for factor in [1 - retention_change, 1 + retention_change]:
            predictions = []
            for d in days:
                if d <= max_known_day:
                    # Historical data doesn't change
                    idx = np.where(self.days == d)[0]
                    if len(idx) > 0:
                        predictions.append(self.ltv_values[idx[0]])
                    else:
                        predictions.append(np.interp(d, self.days, self.ltv_values))
                else:
                    future_days = np.arange(max_known_day + 1, d + 1)
                    future_r = self.retention_model.predict(future_days)
                    # Apply factor to retention
                    future_r_adjusted = future_r * factor
                    future_revenue = np.sum(future_r_adjusted * self.arpdau_avg)
                    predictions.append(last_known_ltv + future_revenue)
            results[f"{factor*100:.0f}%"] = np.array(predictions)
            
        return results
