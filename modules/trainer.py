import numpy as np
from scipy.linalg import inv
from dataclasses import dataclass

@dataclass
class LinearRegression:
    """A linear regression model.
    """
    def __post_init_(self) -> None:
        self.fitted_coeff = None

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        """Train the model."""
        self.fitted_coeff = (inv(X_train.T @ X_train) @ X_train.T).dot(y_train)
        
    
    def predict(self, X_test: np.array) -> None:
        """Make inference."""
        if self.fitted_coeff is not None:
            self.prediction = X_test.dot(self.fitted_coeff)
        else:
            raise ValueError("Model is not fitted yet.")
