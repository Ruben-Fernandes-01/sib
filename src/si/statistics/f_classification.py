from scipy.stats import f_oneway

from si.data.dataset import Dataset


def f_classification(dataset: Dataset):
    """
    Calcula o F-value e o p-value de cada feature, agrupando
    as amostras por classe.

    Parameters
    ----------
    dataset : Dataset

    Returns
    -------
    F : tuple
        F-value de cada feature.
    p : tuple
        p-value de cada feature.
    """
    classes = dataset.get_classes()

    # agrupa as amostras de X por classe
    groups = [dataset.X[dataset.y == c] for c in classes]

    # scipy.stats.f_oneway compara colunas correspondentes entre os grupos
    F, p = f_oneway(*groups)

    return F, p