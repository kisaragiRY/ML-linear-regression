from numpy.typing import NDArray

from trainer import LinearRegression

import numpy as np


def get_metric(metric_name: str):
    """Get metric from a string."""
    try:
        metric = _METRICS[metric_name]
        return metric
    except KeyError:
        raise ValueError(f"{metric_name} is not a valid metric name.")


def mean_square_error(y_true: NDArray, y_pred: NDArray) -> float:
    """Calculate mean square error."""
    return np.sum((y_pred - y_true)**2)/len(y_pred)

def r2_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Calculate R2 score.

    R2 = 1 - SSE / SST
    where, SSE = ∑yi_true - yi_pred, SST = ∑yi_true - y_true_mean

    """
    sse = sum((y_true - y_pred) ** 2)
    sst = sum((y_true - y_true.mean()) ** 2)

    return  1 - (sse / sst)

def condition_number(X: NDArray) -> float:
    """Calculate the condition number of matrix X.

    Its definition is the largest eigenvalue divided by the smallest of matrix X.
    """
    eigenv = np.linalg.eigvals(X)
    return max(eigenv) / min(eigenv)


def vif(X: NDArray) -> NDArray:
    """Calculate the VIF(variance inflation factor) for the design matrix X.

    Return an array of VIF corresponds to each featurn in the design matrix.

    Parameter
    ----------
    X: NDArray 
        Design matrix.
    """
    lr = LinearRegression()
    n_coeff = X.shape[1]
    out = []
    for i in range(n_coeff):
        X_train = X[: , np.arange(n_coeff) != i]
        y_true = X[:, i]
        lr.fit(X_train , y_true)
        y_pred = lr.predict(X_train)
        out.append(1 / (1 - r2_score(y_true, y_pred)))
    return out

_METRICS = dict(
    mean_square_error = mean_square_error,
    condition_number = condition_number,
    r2_score = r2_score,
    vif = vif
)
