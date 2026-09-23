import numpy as np


def accuracy(y_true, y_pred):
    """Devolve a proporção de amostras bem classificadas."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = np.sum(y_true == y_pred)   # nº de labels bem classificadas
    total = len(y_true)                  # nº total de labels
    return float(correct / total)