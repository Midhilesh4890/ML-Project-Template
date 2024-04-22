from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import pandas as pd

class DensityEstimator:

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names, bandwidth=0.1):
        if not hasattr(dim_reducer, 'inverse_transform'):
            raise ValueError("Dimensionality reducer must have an 'inverse_transform' method.")

        self.data = data
        self.dim_reducer = dim_reducer
        self.feature_names = high_dim_feature_names
        self.bandwidth = bandwidth
        self.distribution_model = None

    def fit_distribution(self):
        """Model the distribution of the dataset using Kernel Density Estimation."""
        self.distribution_model = KernelDensity(bandwidth=self.bandwidth)
        self.distribution_model.fit(self.data)

    def sample_parties(self, n_samples=10):
        """Randomly sample parties from the fitted distribution."""
        samples = self.distribution_model.sample(n_samples)
        return samples

    def map_to_high_dim(self, samples):
        """Map the randomly sampled data back to the original higher dimensional space."""
        high_dim_samples = self.dim_reducer.inverse_transform(samples)
        return high_dim_samples

    def process(self, n_samples=10):
        """Run the complete density estimation and sampling process."""
        self.fit_distribution()
        samples = self.sample_parties(n_samples)
        high_dim_samples = self.map_to_high_dim(samples)
        return high_dim_samples


class GuassianDensityEstimator:

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names, n_components=10, covariance_type='full'):
        if not hasattr(dim_reducer, 'inverse_transform'):
            raise ValueError("Dimensionality reducer must have an 'inverse_transform' method.")

        self.data = data
        self.dim_reducer = dim_reducer
        self.feature_names = high_dim_feature_names
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.distribution_model = None

    def fit_distribution(self):
        """Model the distribution of the dataset using Gaussian Mixture Model."""
        self.distribution_model = GaussianMixture(
            n_components=self.n_components, covariance_type=self.covariance_type)
        self.distribution_model.fit(self.data)

    def sample_parties(self):
        """Randomly sample parties from the fitted distribution."""
        samples, _ = self.distribution_model.sample(10)
        return samples

    def map_to_high_dim(self, samples):
        """Map the randomly sampled data back to the original higher dimensional space."""
        high_dim_samples = self.dim_reducer.inverse_transform(samples)
        return high_dim_samples

    def process(self):
        """Run the complete density estimation and sampling process."""
        self.fit_distribution()
        samples = self.sample_parties()
        high_dim_samples = self.map_to_high_dim(samples)
        return high_dim_samples
