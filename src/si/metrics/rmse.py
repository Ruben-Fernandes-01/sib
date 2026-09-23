import numpy as np


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error entre y_true e y_pred.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    return float(np.sqrt(np.sum((y_true - y_pred) ** 2) / n))