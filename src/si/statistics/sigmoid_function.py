import numpy as np


def sigmoid_function(X):
    """Calcula a função sigmóide: probabilidade dos valores serem 1."""
    return 1 / (1 + np.exp(-X))