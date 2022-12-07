from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class WineDataset:
    """Dataset for wine quality data."""
    data_dir : Path

    def __post_init__(self) -> None:
        self.data = pd.read_csv(self.data_dir)

    def split_data(self, train_ratio: float):
        """Split data into train and test set."""
        X_train = self.data.drop("quality", axis=1).sample(frac=train_ratio, random_state=20221207)
        X_test = self.data.drop("quality", axis=1).drop(X_train.index)

        y_train = self.data["quality"].loc[X_train.index]
        y_test = self.data["quality"].loc[X_test.index]
        return X_train, X_test, y_train, y_test 
