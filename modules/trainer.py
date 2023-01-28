from numpy.typing import NDArray

import numpy as np
from scipy.linalg import inv
from dataclasses import dataclass

@dataclass
class LinearRegression:
    """A linear regression model.
    """
    def __post_init_(self) -> None:
        """Initiate.
        """
        self.fitted_coeff = None

    def fit(self, X_train: NDArray, y_train: NDArray, k: float = 0) -> None:
        """Train the model.
        
        Parameter
        ---
        X_train: NDArray
            the design matrix from the training set.
        y_train: NDArray
            the response variable from the training set.
        k: float
            the Ridge parameter, 0 if the OLS(ordinary least square) is implemented.
        """
        self.__is_valid_k(k)
        n_coeff = X_train.shape[1]

        if k == 0:
            try:
                inv(X_train.T @ X_train)
                self.fitted_coeff = (inv(X_train.T @ X_train) @ X_train.T).dot(y_train)
            except:
                raise ValueError("Design matrix is singular.")
        else:
            self.fitted_coeff = (inv(X_train.T @ X_train + k * np.identity(n_coeff)) @ X_train.T).dot(y_train)

        
    def __is_valid_k(self, k: float):
        """Check if the Ridge parameter k is valid.
        """
        if k < 0:
            raise ValueError("Ridge parameter k must be larger than 0.")

    def predict(self, X_test: NDArray) -> None:
        """Make inference.
        
        Parameter
        ---
        X_test: NDArray
            Design matrix from the test set.
        """
        if self.fitted_coeff is not None:
            return X_test.dot(self.fitted_coeff)
        else:
            raise ValueError("Model is not fitted yet.")
