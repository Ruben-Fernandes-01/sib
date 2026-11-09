import numpy as np

from si.data.dataset import Dataset
from si.neural_networks.losses import LossFunction, MeanSquaredError
from si.neural_networks.optimizers import Optimizer, SGD


class NeuralNetwork:
    """
    Rede neuronal que liga as camadas, a loss e o optimizer.

    Parameters
    ----------
    epochs : int
        Número de épocas de treino.
    batch_size : int
        Tamanho dos mini-batches.
    optimizer : classe Optimizer
        Classe do optimizer a usar (ex.: SGD). É instanciada aqui com learning_rate e kwargs.
    learning_rate : float
        Taxa de aprendizagem.
    verbose : bool
        Se True, imprime a loss e a métrica em cada época.
    loss : classe LossFunction
        Classe da loss a usar (ex.: MeanSquaredError). É instanciada aqui.
    metric : callable
        Função metric(y_true, y_pred) a calcular em cada época.
    **kwargs
        Argumentos extra passados ao optimizer (ex.: momentum=0.9).

    Attributes
    ----------
    layers : list
        Camadas da rede.
    history : dict
        Loss e métrica de cada época: {epoch: {'loss': ..., 'metric': ...}}.
    """

    def __init__(self, epochs=100, batch_size=128, optimizer=SGD, learning_rate=0.01,
                 verbose=False, loss=MeanSquaredError, metric=None, **kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer(learning_rate=learning_rate, **kwargs)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric
        self.layers = []
        self.history = {}

    def add(self, layer):
        """
        Adiciona uma camada à rede: define a input shape (a partir da camada anterior,
        exceto na primeira, que a traz de fora), inicializa-a com o optimizer
        (se a camada precisar) e junta-a a self.layers.
        """
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def _forward(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def _backward(self, error):
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    @staticmethod
    def _as_2d(y):
        """y 1D (n,) passa a coluna (n, 1), para bater certo com a saída da rede."""
        return y.reshape(-1, 1) if y.ndim == 1 else y

    @staticmethod
    def _match(y, predictions):
        """Para a métrica: se y é 1D e a saída é (n, 1), achata a saída para (n,)."""
        if y.ndim == 1 and predictions.ndim == 2 and predictions.shape[1] == 1:
            return predictions.ravel()
        return predictions

    def fit(self, dataset):
        """Treina a rede (forward + loss derivative + backward em cada mini-batch)."""
        X = dataset.X
        y = self._as_2d(dataset.y)
        n_samples = X.shape[0]

        for epoch in range(1, self.epochs + 1):
            # baralhar os índices e dividir em mini-batches
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch = indices[start:start + self.batch_size]
                X_batch, y_batch = X[batch], y[batch]

                output = self._forward(X_batch, training=True)
                error = self.loss.derivative(y_batch, output)
                self._backward(error)

            # loss e métrica da época, com os dados todos
            predictions = self._forward(X, training=False)
            loss_value = self.loss.loss(y, predictions)
            if self.metric is not None:
                metric_value = self.metric(dataset.y, self._match(dataset.y, predictions))
            else:
                metric_value = "NA"
            self.history[epoch] = {'loss': loss_value, 'metric': metric_value}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss_value:.4f} - metric: {metric_value}")
        return self

    def predict(self, dataset):
        """Previsões da rede (forward propagation em modo de inferência)."""
        return self._forward(dataset.X, training=False)

    def score(self, dataset, predictions=None):
        """Avalia a rede com a métrica."""
        if self.metric is None:
            raise ValueError("A rede foi criada sem metric.")
        if predictions is None:
            predictions = self.predict(dataset)
        return self.metric(dataset.y, self._match(dataset.y, predictions))
