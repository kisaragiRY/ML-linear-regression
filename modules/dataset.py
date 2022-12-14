from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from typing import Tuple

@dataclass
class WineDataset:
    """Dataset for wine quality data."""
    data_dir : Path

    def __post_init__(self) -> None:
        self.data = pd.read_csv(self.data_dir).drop("quality", axis=1)

    def split_data(self, train_ratio: float) -> Tuple:
        """Split data into train and test set.
        
        Return
        -----------
        X_train : np.array
            Independent variables from train set.
        X_test : np.array
            Independent variables from test set.
        y_train : np.array
            dependent variables from train set.
        y_test : np.array
            dependent variables from test set.
        """
        X_train = self.data.drop("citric acid", axis=1).sample(frac=train_ratio, random_state=20221207)
        X_test = self.data.drop("citric acid", axis=1).drop(X_train.index)

        y_train = self.data["citric acid"].loc[X_train.index]
        y_test = self.data["citric acid"].loc[X_test.index]
        return X_train.values, X_test.values, y_train.values, y_test.values
