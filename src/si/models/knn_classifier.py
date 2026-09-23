from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Distância euclidiana entre uma amostra x e várias amostras y.

    Parameters
    ----------
    x : np.ndarray
        Amostra única, shape (n_features,).
    y : np.ndarray
        Várias amostras, shape (n_amostras, n_features).

    Returns
    -------
    np.ndarray
        Array com shape (n_amostras,), com a distância de x a cada linha de y.
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1))


class KNNClassifier(Model):
    """
    K-Nearest Neighbors para classificação.

    Estima a classe de uma amostra com base nas k amostras
    mais semelhantes do dataset de treino.

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

    def _fit(self, dataset: Dataset) -> "KNNClassifier":
        """Guarda o dataset de treino."""
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray):
        """Classe mais comum entre os k vizinhos mais próximos de uma amostra."""
        # 1. distância entre a amostra e todas as amostras de treino
        distances = self.distance(sample, self.dataset.X)

        # 2. índices dos k vizinhos mais próximos (menor distância)
        k_nearest_indexes = np.argsort(distances)[:self.k]

        # 3. classes correspondentes em y
        k_nearest_labels = self.dataset.y[k_nearest_indexes]

        # 4. classe com maior frequência
        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """Estima a classe de cada amostra do dataset de teste."""
        # 5. aplica os passos 1-4 a todas as amostras
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset, predictions: np.ndarray = None) -> float:
        """Accuracy entre as classes estimadas e as reais."""
        if predictions is None:
            predictions = self._predict(dataset)
        return accuracy(dataset.y, predictions)