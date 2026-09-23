from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distância euclidiana entre uma amostra x e várias amostras y."""
    return np.sqrt(((x - y) ** 2).sum(axis=1))


class KNNRegressor(Model):
    """
    K-Nearest Neighbors para regressão.

    Estima o valor de uma amostra com base na média dos k
    valores mais semelhantes do dataset de treino.

    Parameters
    ----------
    k : int
        Número de vizinhos mais próximos a considerar.
    distance : Callable
        Função que calcula a distância entre uma amostra e as
        amostras do dataset de treino.

    Attributes
    ----------
    dataset : Dataset
        Dataset de treino (guardado no fit).
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> "KNNRegressor":
        """Guarda o dataset de treino."""
        self.dataset = dataset
        return self

    def _get_closest_value(self, sample: np.ndarray):
        """Média dos valores dos k vizinhos mais próximos de uma amostra."""
        # 1. distâncias
        distances = self.distance(sample, self.dataset.X)

        # 2. índices dos k vizinhos mais próximos
        k_nearest_indexes = np.argsort(distances)[:self.k]

        # 3. valores correspondentes em y
        k_nearest_values = self.dataset.y[k_nearest_indexes]

        # 4. média dos valores
        return np.mean(k_nearest_values)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """Estima o valor de cada amostra do dataset de teste."""
        # 5. aplica os passos 1-4 a todas as amostras
        return np.apply_along_axis(self._get_closest_value, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset, predictions: np.ndarray = None) -> float:
        """RMSE entre os valores estimados e os reais."""
        if predictions is None:
            predictions = self._predict(dataset)
        return rmse(dataset.y, predictions)