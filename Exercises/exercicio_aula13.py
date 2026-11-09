import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.model_selection.grid_search import grid_search_cv
from si.neural_networks.activation import ReLUActivation, SigmoidActivation
from si.neural_networks.layers import DenseLayer
from si.neural_networks.losses import BinaryCrossEntropy
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.optimizers import SGD


# a saída da sigmoid é uma probabilidade (n, 1): achatar e arredondar para 0/1
acc = lambda y_true, y_pred: accuracy(np.ravel(y_true), np.round(np.ravel(y_pred)))


def build_network(learning_rate, n_hidden):
    np.random.seed(0)   # mesmos pesos iniciais para todas as configurações
    net = NeuralNetwork(epochs=100, batch_size=16, optimizer=SGD,
                        learning_rate=learning_rate, momentum=0.9,
                        loss=BinaryCrossEntropy,
                        metric=acc)
    net.add(DenseLayer(n_hidden, (2,)))
    net.add(ReLUActivation())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())
    return net


rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(400, 2))
y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1.5).astype(float)   # dentro do círculo?
ds = Dataset(X, y)

grid = {'learning_rate': [0.001, 0.01, 0.1], 'n_hidden': [2, 4, 16]}
results = grid_search_cv(build_network, ds, grid, cv=5, seed=1)
for h, s in zip(results['hyperparameters'], results['scores']):
    print(h, round(s, 3))
print(results['best_hyperparameters'], results['best_score'])
