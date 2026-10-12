import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):
    """
    Regressão logística com regularização L2, resolvida por Gradient Descent.

    Parameters
    ----------
    l2_penalty : float
        Parâmetro de regularização L2 (lambda).
    alpha : float
        Learning rate.
    max_iter : int
        Número máximo de iterações.
    patience : int
        Número máximo de iterações seguidas sem melhoria do custo.
    scale : bool
        Se True, normaliza os dados (média 0, desvio-padrão 1).

    Attributes
    ----------
    theta : np.ndarray
        Coeficientes do modelo (um por feature).
    theta_zero : float
        Coeficiente de intercepção.
    mean : np.ndarray
        Média de cada feature.
    std : np.ndarray
        Desvio-padrão de cada feature.
    cost_history : dict
        Custo em cada iteração {iteração: custo}.
    """

    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001,
                 max_iter: int = 2000, patience: int = 100,
                 scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> "LogisticRegression":
        """Estima theta, theta_zero, mean, std e cost_history."""
        # 1. escala os dados, se necessário
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            self.std[self.std == 0] = 1          # evita divisão por zero
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = X.shape
        y = dataset.y

        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}

        no_improvement = 0

        for i in range(self.max_iter):
            # 2. estima Y com a sigmoid_function
            y_pred = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)
            error = y_pred - y

            # 3. gradiente com a learning rate
            gradient = (self.alpha / m) * np.dot(error, X)

            # 4. termo de regularização L2
            penalization_term = self.theta * (1 - self.alpha * self.l2_penalty / m)

            # 5. atualiza theta
            self.theta = penalization_term - gradient

            # 6. atualiza theta_zero (sem penalização)
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(error)

            # 7. custo
            self.cost_history[i] = self.cost(dataset)

            # 8. paragem por patience
            if i > 0 and self.cost_history[i] >= self.cost_history[i - 1]:
                no_improvement += 1
            else:
                no_improvement = 0

            if no_improvement >= self.patience:
                break

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """Prevê Y (0 ou 1) usando theta, theta_zero e a sigmoid_function."""
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X

        # 1. prevê as probabilidades
        probabilities = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)

        # 2. converte as probabilidades em binário
        return np.where(probabilities >= 0.5, 1, 0)

    def _score(self, dataset: Dataset, predictions: np.ndarray = None) -> float:
        """Accuracy entre as classes reais e as previstas."""
        if predictions is None:
            predictions = self._predict(dataset)
        return accuracy(dataset.y, predictions)

    def cost(self, dataset: Dataset) -> float:
        """Função de custo J(theta): log-loss com regularização L2."""
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        m = dataset.X.shape[0]

        predictions = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)   # evita log(0)

        cost = (-dataset.y * np.log(predictions)
                - (1 - dataset.y) * np.log(1 - predictions))
        cost = np.sum(cost) / m
        cost = cost + (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)
        return float(cost)