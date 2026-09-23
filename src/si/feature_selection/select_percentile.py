import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Seleciona uma percentagem das features com maior F-value.

    Parameters
    ----------
    score_func : Callable
        Função que calcula F e p para cada feature.
    percentile : float
        Percentagem de features a selecionar (ex.: 40 para 40%).

    Attributes
    ----------
    F : np.ndarray
        F-value de cada feature.
    p : np.ndarray
        p-value de cada feature.
    """

    def __init__(self, score_func=f_classification, percentile: float = 50, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> "SelectPercentile":
        """Calcula F e p para cada feature."""
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Seleciona a percentagem de features com maior F-value.
        Trata empates no threshold para manter o nº correto de features.
        """
        n_features = len(self.F)
        n_select = int(n_features * (self.percentile / 100))

        # threshold: valor de F no percentil (100 - percentile)
        threshold = np.percentile(self.F, 100 - self.percentile)

        # começa por selecionar tudo o que é estritamente maior que o threshold
        mask = self.F > threshold
        n_selected = np.sum(mask)

        # se faltarem features (por causa de empates no threshold), acrescenta-as
        if n_selected < n_select:
            tie_idxs = np.where(self.F == threshold)[0]
            n_needed = n_select - n_selected
            mask[tie_idxs[:n_needed]] = True

        idxs = np.where(mask)[0]
        idxs = np.sort(idxs)

        new_X = dataset.X[:, idxs]
        new_features = np.array(dataset.features)[idxs].tolist() if dataset.features else None

        return Dataset(new_X, dataset.y, features=new_features, label=dataset.label)