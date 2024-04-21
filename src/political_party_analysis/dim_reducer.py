import pandas as pd
from sklearn.decomposition import PCA

class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(self, data: pd.DataFrame, n_components: int = 2):
        self.n_components = n_components
        self.data = data
        self.feature_columns = data.columns
        self.model=PCA(n_components=self.n_components)

    # def reduce_dimensions(self):
    #     """Reduce dimensions of the data to the number of components specified."""
    #     # Fit the PCA model and transform the data to reduce its dimensions
    #     reduced_data = self.model.fit_transform(self.data[self.feature_columns])
    #     return reduced_data
    
    def reduce_dimensions(self):
        """Reduce dimensions of the data to the number of components specified."""
        # Fit the PCA model and transform the data to reduce its dimensions
        reduced_data = self.model.fit_transform(self.data[self.feature_columns])
        # Convert the numpy array back to DataFrame with appropriate column names
        reduced_df = pd.DataFrame(reduced_data, columns=[
                                f"Component {i+1}" for i in range(self.n_components)])
        return reduced_df


    def inverse_transform(self, data):
        """Transform data back to the original space."""
        return self.model.inverse_transform(data)
