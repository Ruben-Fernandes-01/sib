import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):
    """
    Seleciona features cuja variância é maior que um threshold.

    Parameters
    ----------
    threshold : float
        Valor de corte (cut-off).

    Attributes
    ----------
    variance : np.ndarray
        Variância de cada feature (estimada no fit).
    """

    def __init__(self, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset: Dataset) -> "VarianceThreshold":
        """Estima a variância de cada feature."""
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """Seleciona as features com variância maior que o threshold."""
        mask = self.variance > self.threshold

        new_X = dataset.X[:, mask]
        new_features = np.array(dataset.features)[mask].tolist() if dataset.features else None

        return Dataset(new_X, dataset.y, features=new_features, label=dataset.label)