import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset, test_size=0.2, random_state=42):
    """
    Divide um Dataset em treino e teste.

    Parameters
    ----------
    dataset : Dataset
        Dataset a dividir.
    test_size : float
        Proporção do dataset para teste (ex.: 0.2 -> 20%).
    random_state : int
        Semente para gerar as permutações.

    Returns
    -------
    (Dataset, Dataset)
        Tuplo (train, test).
    """
    # Semente para os resultados serem reprodutíveis
    np.random.seed(random_state)

    # Número de amostras em cada conjunto
    n_samples = dataset.X.shape[0]
    n_test = int(n_samples * test_size)

    # Permutação aleatória dos índices
    permutations = np.random.permutation(n_samples)

    # Os primeiros n_test índices vão para teste, os restantes para treino
    test_idxs = permutations[:n_test]
    train_idxs = permutations[n_test:]

    train = Dataset(
        dataset.X[train_idxs],
        dataset.y[train_idxs] if dataset.y is not None else None,
        features=dataset.features,
        label=dataset.label,
    )
    test = Dataset(
        dataset.X[test_idxs],
        dataset.y[test_idxs] if dataset.y is not None else None,
        features=dataset.features,
        label=dataset.label,
    )

    return train, test

def stratified_train_test_split(dataset, test_size=0.2, random_state=42):
    """
    Divide um Dataset em treino e teste, mantendo a proporção de classes.

    Parameters
    ----------
    dataset : Dataset
    test_size : float
    random_state : int

    Returns
    -------
    (Dataset, Dataset)
        Tuplo (train, test).
    """
    np.random.seed(random_state)

    # classes únicas e respetivas contagens
    labels, counts = np.unique(dataset.y, return_counts=True)

    train_idxs = []
    test_idxs = []

    for label, count in zip(labels, counts):
        # índices das amostras desta classe
        label_idxs = np.where(dataset.y == label)[0]

        # nº de amostras de teste para esta classe
        n_test = int(count * test_size)

        # baralha e seleciona os índices de teste
        shuffled = np.random.permutation(label_idxs)
        test_idxs.extend(shuffled[:n_test])
        train_idxs.extend(shuffled[n_test:])

    train_idxs = np.array(train_idxs)
    test_idxs = np.array(test_idxs)

    train = Dataset(
        dataset.X[train_idxs],
        dataset.y[train_idxs],
        features=dataset.features,
        label=dataset.label,
    )
    test = Dataset(
        dataset.X[test_idxs],
        dataset.y[test_idxs],
        features=dataset.features,
        label=dataset.label,
    )

    return train, test