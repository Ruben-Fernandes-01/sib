from abc import ABCMeta, abstractmethod

import numpy as np


class Layer(metaclass=ABCMeta):
    """
    Classe base, abstrata, para todas as camadas de uma rede neuronal.
    """

    @abstractmethod
    def forward_propagation(self, input, training):
        """Calcula a saída da camada dado o input."""
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        """Devolve a shape da saída da camada."""
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        """Devolve o número de parâmetros da camada."""
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        """Define a shape do input da camada."""
        self._input_shape = input_shape

    def input_shape(self):
        """Devolve a shape do input da camada."""
        return self._input_shape

    def layer_name(self):
        """Devolve o nome da camada."""
        return self.__class__.__name__


class DenseLayer(Layer):
    """

    Parameters
    ----------
    n_units : int
    input_shape : tuple
    """

    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self._input_shape = input_shape
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

        """

        Parameters
        ----------
        """
        n_features = self.input_shape()[0]
        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))
        return self

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input, training=False):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def output_shape(self):
        return (self.n_units,)