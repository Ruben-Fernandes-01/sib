from abc import abstractmethod

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Classe base, abstrata, para as camadas de ativação de uma rede neuronal.
    Introduzem não-linearidade: não têm parâmetros treináveis, não alteram a
    shape dos dados, aplicam apenas uma função elemento a elemento.
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input, training=False):
        """Guarda o input, aplica a função de ativação e guarda/devolve a saída."""
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        """
        dE/dX = derivada(X) * dE/dY, multiplicação elemento a elemento (não np.dot):
        cada saída desta camada só depende da entrada correspondente.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        """Calcula a função de ativação sobre o input."""
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        """Calcula a derivada da função de ativação sobre o input."""
        raise NotImplementedError

    def output_shape(self):
        """A ativação não muda a shape dos dados."""
        return self.input_shape()

    def parameters(self):
        """As camadas de ativação não têm parâmetros treináveis."""
        return 0


class ReLUActivation(ActivationLayer):
    """ReLU: devolve o input se for positivo, 0 caso contrário."""

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(input > 0, 1, 0)


class SigmoidActivation(ActivationLayer):
    """Sigmoid: comprime o input para o intervalo (0, 1)."""

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        s = self.activation_function(input)
        return s * (1 - s)


class TanhActivation(ActivationLayer):
    """Tangente hiperbólica: comprime o input para o intervalo (-1, 1)."""

    def activation_function(self, input):
        return np.tanh(input)

    def derivative(self, input):
        t = self.activation_function(input)
        return 1 - t ** 2


class SoftmaxActivation(ActivationLayer):
    """
    Softmax: transforma os valores de saída numa distribuição de probabilidades
    (soma 1), usada tipicamente na última camada em classificação multi-classe.

    Versão numericamente estável: subtrai o máximo de cada linha antes do
    exponencial, para evitar exponenciais muito grandes (overflow).
    """

    def activation_function(self, input):
        shifted = input - np.max(input, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def derivative(self, input):
        s = self.activation_function(input)
        return s * (1 - s)