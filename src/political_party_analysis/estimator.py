from sklearn.mixture import GaussianMixture
import pandas as pd

class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """

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
