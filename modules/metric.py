import numpy as np

def get_metric(metric_name: str):
    """Get metric from a string."""
    try:
        metric = _METRICS[metric_name]
        return metric
    except KeyError:
        raise ValueError(f"{metric_name} is not a valid metric name.")

def mean_square_error(y_test, y_pred) -> float:
    """Calculate mean square error."""
    return np.sum((y_pred - y_test)**2)/len(y_pred)

_METRICS = dict(
    mean_square_error = mean_square_error
)
