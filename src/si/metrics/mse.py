import numpy as np


def mse(y_true, y_pred):
    """Mean Squared Error entre y_true e y_pred."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))