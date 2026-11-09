from abc import ABCMeta, abstractmethod

import numpy as np


class LossFunction(metaclass=ABCMeta):
    """
    Classe base, abstrata, para todas as funções de perda (loss) de uma rede neuronal.
    """

    @abstractmethod
    def loss(self, y_true, y_pred):
        """Calcula a loss entre as classes/valores reais e os previstos."""
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        """Calcula a derivada da loss em relação às previsões (dE/dŷ)."""
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """Erro quadrático médio: L = 1/n * soma (y - y*)^2  (usado em regressão)."""

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        # dL/dY = 2/n * (Y* - Y)
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """Entropia cruzada binária: L = - soma (y log(y*) + (1-y) log(1-y*))."""

    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        # dL/dY = -Y/Y* + (1-Y)/(1-Y*)
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / p + (1 - y_true) / (1 - p)


class CategoricalCrossEntropy(LossFunction):
    """
    Entropia cruzada categórica (multi-classe, y em one-hot):
    L = - soma (y log(y*)).
    """

    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p))

    def derivative(self, y_true, y_pred):
        # dL/dY = -Y/Y*
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / p
