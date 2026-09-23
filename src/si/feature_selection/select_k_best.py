import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    Seleciona as k features com maior F-value.

    Parameters
    ----------
    score_func : Callable
        Função que calcula F e p para cada feature.
    k : int
        Número de features a selecionar.

    Attributes
    ----------
    F : np.ndarray
        F-value de cada feature.
    p : np.ndarray
        p-value de cada feature.
    """

    def __init__(self, score_func=f_classification, k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> "SelectKBest":
        """Calcula F e p para cada feature."""
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """Seleciona as k features com maior F-value."""
        idxs = np.argsort(self.F)[-self.k:]     # índices dos k maiores F
        idxs = np.sort(idxs)                     # mantém a ordem original das colunas

        new_X = dataset.X[:, idxs]
        new_features = np.array(dataset.features)[idxs].tolist() if dataset.features else None

        return Dataset(new_X, dataset.y, features=new_features, label=dataset.label)