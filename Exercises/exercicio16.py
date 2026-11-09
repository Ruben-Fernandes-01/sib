import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.neural_networks.activation import ReLUActivation, SigmoidActivation
from si.neural_networks.layers import DenseLayer
from si.neural_networks.losses import BinaryCrossEntropy
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.optimizers import SGD


def build_network(n_features):
    """32 -> 16 -> 8 -> 1 (cada Dense reduz as unidades para metade, exceto a última)."""
    # a saída da sigmoid é uma probabilidade: arredondar para 0/1 antes da accuracy
    metric = lambda y_true, y_pred: accuracy(y_true, np.round(y_pred))
    net = NeuralNetwork(epochs=100, batch_size=16, optimizer=SGD, learning_rate=0.01,
                        loss=BinaryCrossEntropy, metric=metric)
    net.add(DenseLayer(n_features // 2, (n_features,)))
    net.add(ReLUActivation())
    net.add(DenseLayer(n_features // 4))
    net.add(ReLUActivation())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())
    return net


def k_fold_scores(dataset, k=5, seed=42):
    """k-fold cross validation simples (troca pela tua função da aula 8, se preferires)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset.X.shape[0])
    folds = np.array_split(indices, k)
    scores = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        train = Dataset(dataset.X[train_idx], dataset.y[train_idx])
        test = Dataset(dataset.X[test_idx], dataset.y[test_idx])
        net = build_network(dataset.X.shape[1])
        net.fit(train)
        scores.append(net.score(test))
    return scores


if __name__ == "__main__":
    # dados sintéticos: 32 features, classificação binária
    np.random.seed(0)
    X = np.random.randn(300, 32)
    y = (X[:, :5].sum(axis=1) > 0).astype(float)
    ds = Dataset(X, y)

    scores = k_fold_scores(ds, k=5)
    print("scores por fold:", np.round(scores, 3))
    print(f"accuracy média: {np.mean(scores):.3f} +- {np.std(scores):.3f}")
