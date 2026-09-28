import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Regressão linear com regularização L2, resolvida por mínimos quadrados
    (solução analítica).

    Parameters
    ----------
    l2_penalty : float
        Parâmetro de regularização L2 (lambda).
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
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> "RidgeRegressionLeastSquares":
        """Estima theta, theta_zero, mean e std."""
        # 1. escala os dados, se necessário
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            self.std[self.std == 0] = 1          # evita divisão por zero
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # 2. acrescenta a coluna de uns (intercepção) na primeira posição
        X = np.c_[np.ones(X.shape[0]), X]

        # 3. matriz de penalização: l2_penalty * matriz identidade
        penalty_matrix = self.l2_penalty * np.eye(X.shape[1])

        # 4. a intercepção (theta_zero) não é penalizada
        penalty_matrix[0, 0] = 0

        # 5. parâmetros do modelo: theta = (X^T X + lambda*P)^-1 X^T y
        thetas = np.linalg.inv(X.T.dot(X) + penalty_matrix).dot(X.T).dot(dataset.y)
        self.theta_zero = thetas[0]
        self.theta = thetas[1:]

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """Prevê Y usando theta e theta_zero (com mean e std do fit)."""
        # 1. escala os dados, se necessário (com mean e std do fit)
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X

        # 2. acrescenta a coluna de uns
        X = np.c_[np.ones(X.shape[0]), X]

        # 3. Y previsto = X . [theta_zero, theta]
        thetas = np.r_[self.theta_zero, self.theta]
        return X.dot(thetas)

    def _score(self, dataset: Dataset, predictions: np.ndarray = None) -> float:
        """MSE entre os valores reais e os previstos."""
        if predictions is None:
            predictions = self._predict(dataset)
        return mse(dataset.y, predictions)