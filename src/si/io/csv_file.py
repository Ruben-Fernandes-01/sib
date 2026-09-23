import numpy as np
import pandas as pd

# Ajusta o import à estrutura do teu projeto
from si.data.dataset import Dataset


def read_csv(filename, sep=',', features=False, label=False):
    """Lê um ficheiro CSV e devolve um Dataset."""
    # header=0 -> 1.ª linha tem os nomes das features; header=None -> não tem
    data = pd.read_csv(filename, sep=sep, header=0 if features else None)

    if label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        feature_names = list(data.columns[:-1]) if features else None
        label_name = data.columns[-1] if features else None
    else:
        X = data.to_numpy()
        y = None
        feature_names = list(data.columns) if features else None
        label_name = None

    return Dataset(X, y, feature_names, label_name)


def write_csv(filename, dataset, sep=',', features=False, label=False):
    """Escreve um Dataset num ficheiro CSV."""
    data = pd.DataFrame(dataset.X)

    if features and dataset.features is not None:
        data.columns = dataset.features

    if label and dataset.y is not None:
        if features:
            y_name = dataset.label or "y"
        else:
            y_name = data.shape[1]
        data[y_name] = dataset.y

    data.to_csv(filename, sep=sep, index=False, header=features)


