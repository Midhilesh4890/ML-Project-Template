from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pdb

class DataLoader:
    """Class to load the political parties dataset"""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = []
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        data_path, _ = urlretrieve(
            self.data_url,
            Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"]),
        )
        return pd.read_stata(data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        df = df.drop_duplicates()
        return df

    def remove_nonfeature_cols(
        self, df: pd.DataFrame, non_features: List[str], index: List[str]
    ) -> pd.DataFrame:
        """Write a function to remove certain features cols and set certain cols as indices
        in a dataframe"""
        df = df.drop(columns=non_features)
        df = df.set_index(index)
        return df

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to handle NaN values in a dataframe"""
        return df.fillna(df.mean())

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to normalise values in a dataframe. Use StandardScaler."""
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])
        return df

    def preprocess_data(self):
        """Write a function to combine all pre-processing steps for the dataset"""
        df = self.party_data
        df.to_csv('ches.csv')
        new_df = pd.read_csv('ches.csv')
        #df = self.remove_duplicates(df)
        df = self.remove_nonfeature_cols(df, self.non_features, self.index)
        df = self.handle_NaN_values(df)
        df = self.scale_features(df)
        return df
