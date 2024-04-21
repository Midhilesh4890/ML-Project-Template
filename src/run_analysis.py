from pathlib import Path
import pandas as pd
from matplotlib import pyplot
import numpy as np

from political_party_analysis.loader import DataLoader
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator
from political_party_analysis.visualization import scatter_plot, plot_density_estimation_results, plot_finnish_parties

if __name__ == "__main__":

    data_loader = DataLoader()
    preprocessed_data = data_loader.preprocess_data()

    # Reset index to turn MultiIndex into columns
    preprocessed_data = preprocessed_data.reset_index()

    # Now you can directly access 'party' and 'country' as columns
    categorical_data = preprocessed_data[['party', 'country']].copy()

    # Dimensionality reduction step (exclude non-numeric columns before PCA)
    numeric_columns = preprocessed_data.select_dtypes(include=[np.number]).columns
    dim_reducer = DimensionalityReducer(data=preprocessed_data[numeric_columns])
    reduced_dim_data = dim_reducer.reduce_dimensions()

    # Merge the reduced dimensionality data with categorical data
    reduced_dim_data = pd.DataFrame(reduced_dim_data, columns=[
                                    f"Component {i+1}" for i in range(dim_reducer.n_components)])
    reduced_dim_data = pd.concat(
        [reduced_dim_data, categorical_data.reset_index(drop=True)], axis=1)
    
    # Uncomment this snippet to plot dim reduced data
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))

    # Density estimation/distribution modelling step
    density_estimator = DensityEstimator(
        # ensure only numeric data is used here
        data=reduced_dim_data[['Component 1', 'Component 2']],
        dim_reducer=dim_reducer,
        high_dim_feature_names=preprocessed_data.columns
    )
    density_estimator.fit_distribution()
    samples = density_estimator.sample_parties()
    high_dim_samples = density_estimator.map_to_high_dim(samples)

    # Plot density estimation results here
    plot_density_estimation_results(reduced_dim_data, samples[:, 0], density_estimator.distribution_model.means_,
                                    density_estimator.distribution_model.covariances_, "Density Estimation")
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))

    # Plot left and right wing parties here
    pyplot.figure()
    splot = pyplot.subplot()
    plot_finnish_parties(reduced_dim_data, splot=splot)
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    pyplot.title("Lefty/righty parties")

    # Plot finnish parties here
    ##### YOUR CODE GOES HERE #####

    print("Analysis Complete")
