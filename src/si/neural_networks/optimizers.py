from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(metaclass=ABCMeta):
    """
    Classe base, abstrata, para os optimizers de uma rede neuronal.

    Parameters
    ----------
    learning_rate : float
        Taxa de aprendizagem do optimizer.
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w, grad_loss_w):
        """
        Atualiza os parâmetros (pesos ou biases) de uma camada.

        Parameters
        ----------
        w : np.ndarray
            Parâmetros atuais da camada.
        grad_loss_w : np.ndarray
            Gradiente da loss em relação a esses parâmetros.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent, com momentum opcional.

    Parameters
    ----------
    learning_rate : float
        Taxa de aprendizagem.
    momentum : float
        Momentum (beta). Com 0 é o gradiente descendente simples.

    Attributes
    ----------
    retained_gradient : np.ndarray
        Gradiente acumulado das iterações anteriores.
    """

    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w, grad_loss_w):
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        # V_t = beta * V_{t-1} + (1 - beta) * dL/dW
        self.retained_gradient = (self.momentum * self.retained_gradient
                                  + (1 - self.momentum) * grad_loss_w)
        # W_t = W_{t-1} - alpha * V_t
        return w - self.learning_rate * self.retained_gradient


class Adam(Optimizer):
    """
    Adam: combina o RMSprop com o SGD com momentum (médias móveis do gradiente
    e do gradiente ao quadrado, com correção de enviesamento).

    Parameters
    ----------
    learning_rate : float
        Taxa de aprendizagem.
    beta_1 : float
        Taxa de decaimento da estimativa do 1.º momento.
    beta_2 : float
        Taxa de decaimento da estimativa do 2.º momento.
    epsilon : float
        Constante pequena para estabilidade numérica.

    Attributes
    ----------
    m, v : np.ndarray
        Médias móveis do gradiente e do gradiente ao quadrado.
    t : int
        Contador de atualizações (time stamp).
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad_loss_w):
        if self.m is None or self.v is None:
            self.m = np.zeros(np.shape(w))
            self.v = np.zeros(np.shape(w))
        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(grad_loss_w, 2)
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
