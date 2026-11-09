import copy
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
    def backward_propagation(self, output_error):
        """Propaga o erro de saída para trás e devolve o erro do input."""
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
    Camada totalmente ligada (fully connected) de uma rede neuronal.

    Parameters
    ----------
    n_units : int
        Número de neurónios/unidades da camada.
    input_shape : tuple
        Shape do input da camada (opcional; pode ser definida depois com set_input_shape).
    """

    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self._input_shape = input_shape
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.w_opt = None
        self.b_opt = None

    def initialize(self, optimizer):
        """
        Inicializa os pesos e biases da camada, e guarda uma cópia do optimizer
        para cada um deles (w_opt para os pesos, b_opt para os biases), para que
        cada parâmetro tenha o seu próprio estado interno do optimizer (ex.: momentum).

        Parameters
        ----------
        optimizer : objeto com um método update(w, grad)
            Optimizer a usar para atualizar os pesos e os biases desta camada.
        """
        n_features = self.input_shape()[0]
        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self):
        """Devolve o número total de parâmetros (pesos + biases) da camada."""
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input, training=False):
        """Calcula Y = X . W + b e guarda o input (necessário para o backward)."""
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        """
        Propagação do erro para trás (backpropagation) desta camada.

        Parameters
        ----------
        output_error : np.ndarray
            dE/dY — erro vindo da camada seguinte (ou da loss, se for a última camada).

        Returns
        -------
        np.ndarray
            dE/dX — erro a passar para a camada anterior (será o dE/dY dela).
        """
        # 1. erro do input, dE/dX = dE/dY . W^T  (calculado ANTES de atualizar os pesos!)
        input_error = np.dot(output_error, self.weights.T)
        # 2. erro dos pesos, dE/dW = X^T . dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # 3. erro dos biases, dE/dB = soma de dE/dY ao longo do batch
        bias_error = np.sum(output_error, axis=0, keepdims=True)
        # 4. e 5. atualizar pesos e biases com os respetivos optimizers
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        # 6. devolver o erro do input, calculado no passo 1
        return input_error

    def output_shape(self):
        """Devolve a shape da saída da camada: (n_units,)."""
        return (self.n_units,)


class Dropout(Layer):
    """
    Camada de regularização Dropout: durante o treino, "desliga" aleatoriamente
    uma fração das entradas (definida por probability), ajudando a evitar overfitting.
    Em inferência, não faz nada (devolve o input tal e qual).

    Parameters
    ----------
    probability : float
        Taxa de dropout, entre 0 e 1 (fração de unidades a desligar).
    """

    def __init__(self, probability):
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input, training):
        """
        Em modo de treino, gera uma máscara binomial (0/1) com probabilidade
        (1 - probability) de manter cada unidade, aplica-a ao input e reescala
        pelo fator 1 / (1 - probability) para manter a magnitude esperada do sinal.
        Em modo de inferência, devolve o input sem alterações.
        """
        self.input = input
        if training:
            scaling = 1 / (1 - self.probability)
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)
            self.output = input * self.mask * scaling
        else:
            self.output = input
        return self.output

    def backward_propagation(self, output_error):
        """Propaga o erro só pelas unidades que ficaram ativas (multiplica pela máscara)."""
        return output_error * self.mask

    def output_shape(self):
        """O dropout não muda a shape dos dados."""
        return self.input_shape()

    def parameters(self):
        """O dropout não tem parâmetros treináveis."""
        return 0
