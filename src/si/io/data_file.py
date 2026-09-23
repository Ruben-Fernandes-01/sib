import numpy as np
import pandas as pd

# Ajusta o import à estrutura do teu projeto
from si.data.dataset import Dataset


def read_data_file(filename, sep=',', label=False):
    """Lê um ficheiro de dados numéricos (sem nomes) e devolve um Dataset."""
    data = np.genfromtxt(filename, delimiter=sep)

    if label:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data
        y = None

    return Dataset(X, y)


def write_data_file(filename, dataset, sep=',', label=False):
    """Escreve um Dataset num ficheiro de dados numéricos (sem nomes)."""
    if label and dataset.y is not None:
        data = np.column_stack((dataset.X, dataset.y))
    else:
        data = dataset.X

    np.savetxt(filename, data, delimiter=sep)