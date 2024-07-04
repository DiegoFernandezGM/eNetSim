import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, lognorm, expon, gamma

class DistributionAnalyzer:
    def __init__(self, data):
        self.data = data

    def fit_distributions(self):
        distributions = [norm, lognorm, expon, gamma]
        init_params = [[0, 1], [0, 1], [1.0], [1, 1]]
        results = []
        for dist in distributions:
            params, _ = curve_fit(dist.pdf, self.data, init_params[distributions.index(dist)])
            results.append((dist, params))
        return results

    def calculate_aic_bic(self, results):
        aic_bic_values = []
        for dist, params in results:
            ll = dist.logpdf(self.data, *params).sum()
            aic = -2 * ll + len(params)
            bic = -2 * ll + len(params) * np.log(len(self.data))
            aic_bic_values.append((dist, aic, bic))
        return aic_bic_values

    def select_best_distribution(self, aic_bic_values):
        best_dist = min(aic_bic_values, key=lambda x: x[1])[0]
        return best_dist

    def sample_from_distribution(self, best_dist):
        if best_dist == norm:
            return np.random.normal(loc=np.mean(self.data), scale=10, size=1000)
        elif best_dist == lognorm:
            return np.random.lognormal(mean=np.mean(self.data), sigma=1, size=1000)
        elif best_dist == expon:
            return np.random.exponential(scale=np.mean(self.data), size=1000)
        elif best_dist == gamma:
            return np.random.gamma(shape=1, scale=np.mean(self.data), size=1000)
        else:
            return np.random.uniform(low=min(self.data), high=max(self.data), size=1000)
        
    def analyze_and_sample(self):
            results = self.fit_distributions()
            aic_bic_values = self.calculate_aic_bic(results)
            best_dist = self.select_best_distribution(aic_bic_values)
            sample = self.sample_from_distribution(best_dist)
            return sample