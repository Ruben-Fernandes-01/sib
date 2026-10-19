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
    Camada totalmente ligada (fully connected / dense layer).
    Liga cada neurónio a todos os neurónios da camada anterior e seguinte.

    Parameters
    ----------
    n_units : int
        Número de neurónios da camada.
    input_shape : tuple
        Shape do input (para dados tabulares, (n_features,)).

    Attributes
    ----------
    input : np.ndarray
        Input da camada.
    output : np.ndarray
        Output da camada.
    weights : np.ndarray
        Pesos da camada.
    biases : np.ndarray
        Bias da camada.
    """

    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self._input_shape = input_shape
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer=None):
        """
        Inicializa os pesos e o bias da camada, e o otimizador.

        Parameters
        ----------
        optimizer
            Otimizador a usar para atualizar os pesos e o bias da camada
            (opcional; ainda não é necessário até à propagação para trás).
        """
        # 1. pesos: distribuição centrada em 0, entre -0.5 e 0.5
        #    matriz (n_features, n_units)
        n_features = self.input_shape()[0]
        self.weights = np.random.rand(n_features, self.n_units) - 0.5

        # 2. bias a zero, matriz (1, n_units)
        self.biases = np.zeros((1, self.n_units))

        # 3. otimizador para os pesos e para o bias
        self.optimizer = optimizer

        # 4. devolve-se a si próprio
        return self

    def parameters(self):
        """Devolve o número de parâmetros da camada (pesos + bias)."""
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input, training=False):
        """
        Propagação para a frente: calcula a saída da camada dado o input.

        Parameters
        ----------
        input : np.ndarray
            Input da camada.
        """
        # guarda o input
        self.input = input

        # calcula e guarda o output: Y = X . W + B
        self.output = np.dot(self.input, self.weights) + self.biases

        return self.output

    def output_shape(self):
        """Devolve a shape da saída da camada."""
        return (self.n_units,)