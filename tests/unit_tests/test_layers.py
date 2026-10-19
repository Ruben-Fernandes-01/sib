import unittest

import numpy as np

from si.neural_networks.layers import DenseLayer, Layer


class TestLayer(unittest.TestCase):

    def test_cannot_instantiate_abstract_layer(self):
        with self.assertRaises(TypeError):
            Layer()

    def test_set_and_get_input_shape(self):
        layer = DenseLayer(4, (3,))

        layer.set_input_shape((5,))

        self.assertEqual((5,), layer.input_shape())

    def test_layer_name(self):
        layer = DenseLayer(4, (3,))

        self.assertEqual("DenseLayer", layer.layer_name())


class TestDenseLayer(unittest.TestCase):

    def test_slide_example(self):
        # exemplo dado no slide "Tests:"
        layer = DenseLayer(4, (3,)).initialize()
        out = layer.forward_propagation(np.random.rand(10, 3))

        self.assertEqual((10, 4), out.shape)
        self.assertEqual(3 * 4 + 4, layer.parameters())
        self.assertEqual((4,), layer.output_shape())

    def test_initialize_weights_shape_and_range(self):
        layer = DenseLayer(5, (4,)).initialize()

        self.assertEqual((4, 5), layer.weights.shape)
        self.assertEqual((1, 5), layer.biases.shape)
        self.assertTrue(np.all(layer.weights >= -0.5))
        self.assertTrue(np.all(layer.weights < 0.5))

    def test_biases_start_at_zero(self):
        layer = DenseLayer(5, (4,)).initialize()

        self.assertTrue(np.all(layer.biases == 0))

    def test_forward_propagation_formula(self):
        layer = DenseLayer(2, (3,)).initialize()
        # fixa pesos e bias conhecidos para verificar a fórmula Y = XW + B
        layer.weights = np.array([[1.0, 0.0],
                                  [0.0, 1.0],
                                  [1.0, 1.0]])
        layer.biases = np.array([[0.5, -0.5]])

        X = np.array([[1.0, 2.0, 3.0]])
        output = layer.forward_propagation(X)

        expected = np.array([[1 * 1 + 2 * 0 + 3 * 1 + 0.5,
                              1 * 0 + 2 * 1 + 3 * 1 - 0.5]])
        self.assertTrue(np.allclose(expected, output))

    def test_forward_propagation_stores_input(self):
        layer = DenseLayer(3, (2,)).initialize()
        X = np.random.rand(7, 2)

        layer.forward_propagation(X)

        self.assertTrue(np.array_equal(X, layer.input))

    def test_initialize_returns_self(self):
        layer = DenseLayer(3, (2,))

        result = layer.initialize()

        self.assertIs(layer, result)


if __name__ == '__main__':
    unittest.main()