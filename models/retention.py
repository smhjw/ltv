import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

class RetentionModel:
    def __init__(self, model_type='weibull'):
        """
        Initialize the retention model.
        :param model_type: 'weibull' or 'lognormal'
        """
        self.model_type = model_type
        self.params = None
        self.cov = None
        self.days = None
        self.retention_rates = None

    def _weibull_func(self, t, lambda_, k):
        """
        Weibull retention function: R(t) = exp(-(t/lambda)^k)
        Note: t should be > 0.
        """
        # Ensure t is float to avoid integer division issues if any
        t = np.array(t, dtype=float)
        return np.exp(-((t / lambda_) ** k))

    def _lognormal_func(self, t, mu, sigma):
        """
        Log-Normal retention function (Survival Function).
        R(t) = 0.5 * erfc((ln(t) - mu) / (sigma * sqrt(2)))
        Using scipy.special.erfc would be better, but implementing roughly here or using scipy.stats.
        Let's use scipy.stats.lognorm.sf but for curve_fit we need a callable with simple params.
        """
        from scipy.special import erfc
        t = np.array(t, dtype=float)
        # Avoid log(0)
        t_safe = np.maximum(t, 1e-10)
        return 0.5 * erfc((np.log(t_safe) - mu) / (sigma * np.sqrt(2)))

    def fit(self, days, retention_rates):
        """
        Fit the model to the provided data.
        :param days: List or array of days (e.g., [1, 2, 3, 7, 14, 30])
        :param retention_rates: List or array of retention rates (0 to 1) (e.g., [0.5, 0.4, ...])
        """
        self.days = np.array(days)
        self.retention_rates = np.array(retention_rates)
        
        # Initial guesses
        if self.model_type == 'weibull':
            # Guess: lambda approx day where ret is 0.36, k approx 0.5
            p0 = [30.0, 0.5]
            bounds = ([0.1, 0.01], [np.inf, 5.0])
            func = self._weibull_func
        elif self.model_type == 'lognormal':
            p0 = [3.0, 1.0] # mu, sigma
            bounds = ([-np.inf, 0.001], [np.inf, 10.0])
            func = self._lognormal_func
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        try:
            self.params, self.cov = curve_fit(func, self.days, self.retention_rates, p0=p0, bounds=bounds)
        except Exception as e:
            raise RuntimeError(f"Model fitting failed: {str(e)}")

    def predict(self, days):
        """
        Predict retention for given days.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet.")
        
        days = np.array(days)
        if self.model_type == 'weibull':
            return self._weibull_func(days, *self.params)
        elif self.model_type == 'lognormal':
            return self._lognormal_func(days, *self.params)

    def get_metrics(self):
        """
        Calculate R2, RMSE, MAPE based on fitted data.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet.")
        
        pred = self.predict(self.days)
        actual = self.retention_rates
        
        r2 = r2_score(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        # Avoid division by zero for MAPE
        mape = np.mean(np.abs((actual - pred) / np.maximum(actual, 1e-10))) * 100
        
        return {
            "R2": r2,
            "RMSE": rmse,
            "MAPE": mape
        }

    def predict_with_interval(self, days, confidence=0.80):
        """
        Predict with confidence intervals using the delta method or simulation.
        Here we use a simple Monte Carlo simulation based on parameter covariance.
        """
        if self.params is None or self.cov is None:
            raise ValueError("Model not fitted yet.")
            
        days = np.array(days)
        pred_mean = self.predict(days)
        
        # Monte Carlo simulation for intervals
        num_simulations = 1000
        simulated_params = np.random.multivariate_normal(self.params, self.cov, num_simulations)
        
        simulated_curves = []
        for params in simulated_params:
            if self.model_type == 'weibull':
                # Ensure positive params
                if params[0] <= 0 or params[1] <= 0: continue
                simulated_curves.append(self._weibull_func(days, *params))
            elif self.model_type == 'lognormal':
                 if params[1] <= 0: continue
                 simulated_curves.append(self._lognormal_func(days, *params))
        
        simulated_curves = np.array(simulated_curves)
        
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        lower_bound = np.percentile(simulated_curves, lower_percentile, axis=0)
        upper_bound = np.percentile(simulated_curves, upper_percentile, axis=0)
        
        return pred_mean, lower_bound, upper_bound

