import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.model_selection.randomized_search import randomized_search_cv, log_uniform
from si.neural_networks.activation import ReLUActivation, SigmoidActivation
from si.neural_networks.layers import DenseLayer
from si.neural_networks.losses import BinaryCrossEntropy
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.optimizers import SGD




# a saída da sigmoid é uma probabilidade (n, 1): achatar e arredondar para 0/1
acc = lambda y_true, y_pred: accuracy(np.ravel(y_true), np.round(np.ravel(y_pred)))


def build_network(learning_rate, batch_size, epochs, n_features=32):
    """Arquitetura do Exercício 16: 32 -> 16 -> 8 -> 1."""
    np.random.seed(0)   # mesmos pesos iniciais para todas as configurações
    net = NeuralNetwork(epochs=int(epochs), batch_size=int(batch_size), optimizer=SGD,
                        learning_rate=learning_rate, loss=BinaryCrossEntropy, metric=acc)
    net.add(DenseLayer(n_features // 2, (n_features,)))
    net.add(ReLUActivation())
    net.add(DenseLayer(n_features // 4))
    net.add(ReLUActivation())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())
    return net


def train_test_split_simple(dataset, test_size=0.2, seed=42):
    """Troca pela tua train_test_split (si.model_selection.split)."""
    idx = np.random.default_rng(seed).permutation(dataset.X.shape[0])
    n_test = int(len(idx) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (Dataset(dataset.X[train_idx], dataset.y[train_idx]),
            Dataset(dataset.X[test_idx], dataset.y[test_idx]))


if __name__ == "__main__":
    # dados sintéticos (32 features, binário) - troca pelo dataset do Exercício 16
    np.random.seed(0)
    X = np.random.randn(300, 32)
    y = (X[:, :5].sum(axis=1) > 0).astype(float)
    ds = Dataset(X, y)

    train, test = train_test_split_simple(ds, test_size=0.2)   # ou train_test_split(ds, test_size=0.2)

    distributions = {
        'learning_rate': log_uniform(1e-4, 1e-1),
        'batch_size': [8, 16, 32, 64],
        'epochs': [50, 100, 200],
    }
    results = randomized_search_cv(build_network, train, distributions,
                                   scoring=acc, cv=3, n_iter=20, seed=1)
    for h, s in zip(results['hyperparameters'], results['scores']):
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in h.items()}, round(s, 3))
    best = results['best_hyperparameters']
    print("\nMelhor:", best, "| score CV:", round(results['best_score'], 3))

    # refit da melhor configuração em todo o treino e avaliação no teste
    final = build_network(**best)
    final.fit(train)
    print("Accuracy no teste (afinada):  ", round(final.score(test), 3))

    # comparação com o Exercício 16 (lr=0.01, batch=16, epochs=100)
    base = build_network(learning_rate=0.01, batch_size=16, epochs=100)
    base.fit(train)
    print("Accuracy no teste (Ex. 16):   ", round(base.score(test), 3))
