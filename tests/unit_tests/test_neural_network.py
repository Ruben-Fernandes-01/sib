import unittest

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.neural_networks.activation import SigmoidActivation, ReLUActivation
from si.neural_networks.layers import DenseLayer
from si.neural_networks.losses import MeanSquaredError, BinaryCrossEntropy
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.optimizers import SGD, Adam


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0], dtype=float)
        self.xor = Dataset(X, y)

    def _xor_net(self, **kwargs):
        np.random.seed(1)
        net = NeuralNetwork(epochs=5000, batch_size=4, optimizer=SGD, learning_rate=0.5,
                            loss=MeanSquaredError, metric=accuracy, **kwargs)
        net.add(DenseLayer(4, (2,)))
        net.add(SigmoidActivation())
        net.add(DenseLayer(1))
        net.add(SigmoidActivation())
        return net

    def test_add_sets_input_shapes_and_initializes(self):
        net = self._xor_net()
        self.assertEqual(len(net.layers), 4)
        self.assertEqual(net.layers[2].input_shape(), (4,))
        self.assertEqual(net.layers[2].weights.shape, (4, 1))
        self.assertIsNotNone(net.layers[0].w_opt)

    def test_optimizer_is_instantiated_with_kwargs(self):
        net = NeuralNetwork(optimizer=SGD, learning_rate=0.3, momentum=0.8)
        self.assertEqual(net.optimizer.learning_rate, 0.3)
        self.assertEqual(net.optimizer.momentum, 0.8)

    def test_predict_shape(self):
        net = self._xor_net()
        self.assertEqual(net.predict(self.xor).shape, (4, 1))

    def test_fit_learns_xor(self):
        net = self._xor_net()
        net.fit(self.xor)
        preds = net.predict(self.xor).ravel()
        np.testing.assert_allclose(preds, [0.05, 0.95, 0.95, 0.07], atol=0.02)

    def test_history_has_one_entry_per_epoch_and_loss_decreases(self):
        net = self._xor_net()
        net.epochs = 200
        net.fit(self.xor)
        self.assertEqual(len(net.history), 200)
        self.assertLess(net.history[200]['loss'], net.history[1]['loss'])

    def test_score_uses_metric(self):
        net = self._xor_net()
        net.fit(self.xor)
        net.metric = lambda y, p: accuracy(y, np.round(p))
        self.assertEqual(net.score(self.xor), 1.0)

    def test_score_without_metric_raises(self):
        net = NeuralNetwork(metric=None)
        net.add(DenseLayer(1, (2,)))
        with self.assertRaises(ValueError):
            net.score(self.xor)

    def test_works_with_adam_and_bce_and_relu(self):
        np.random.seed(0)
        net = NeuralNetwork(epochs=2000, batch_size=4, optimizer=Adam, learning_rate=0.05,
                            loss=BinaryCrossEntropy)
        net.add(DenseLayer(8, (2,)))
        net.add(ReLUActivation())
        net.add(DenseLayer(1))
        net.add(SigmoidActivation())
        net.fit(self.xor)
        np.testing.assert_array_equal(net.predict(self.xor).round().ravel(), [0, 1, 1, 0])

    def test_batch_size_not_dividing_dataset(self):
        net = self._xor_net()
        net.batch_size, net.epochs = 3, 5
        net.fit(self.xor)
        self.assertEqual(len(net.history), 5)


if __name__ == "__main__":
    unittest.main()
