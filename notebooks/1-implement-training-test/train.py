import pickle
import os

from dataset import WineDataset
from trainer import LinearRegression
from param import *

def main():
    """Main."""
    # ---- 1. load the data ----
    dataset = WineDataset(ParamDir().DATA_DIR)
    X_train, X_test, y_train, y_test  = dataset.split_data(ParamTrain().train_ratio)

    # ---- 2. train the model ----
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # ---- 3. test the model ----
    lr.predict(X_test)
    y_pred = lr.prediction

    # ---- 4. save results ----
    results = {
        "y_pred": y_pred,
        "y_test": y_test,
        "estimator": lr
    }

    output_dir = ParamDir().OUTPUT_DIR
    if not output_dir.is_dir():
        os.mkdir(output_dir)
    with open(output_dir/"lr_results.pickle", "wb") as f:
        pickle.dump(results, f)
if __name__ == "__main__":
    main()