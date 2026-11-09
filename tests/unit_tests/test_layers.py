import unittest

import numpy as np

from si.neural_networks.layers import Layer, DenseLayer, Dropout


class NoUpdate:
    def update(self, w, grad):
        return w


class TestLayerBase(unittest.TestCase):

    def test_layer_is_abstract(self):
        with self.assertRaises(TypeError):
            Layer()

    def test_subclass_without_backward_propagation_cannot_be_instantiated(self):
        class Incomplete(Layer):
            def forward_propagation(self, input, training):
                return input

            def output_shape(self):
                return (1,)

            def parameters(self):
                return 0

        with self.assertRaises(TypeError):
            Incomplete()


class TestDenseLayer(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.X = np.random.rand(4, 3)
        self.layer = DenseLayer(2, input_shape=(3,)).initialize(NoUpdate())

    def test_forward_propagation_shape(self):
        out = self.layer.forward_propagation(self.X, training=True)
        self.assertEqual(out.shape, (4, 2))

    def test_output_shape(self):
        self.assertEqual(self.layer.output_shape(), (2,))

    def test_parameters_count(self):
        # 3*2 pesos + 2 biases = 8
        self.assertEqual(self.layer.parameters(), 8)

    def test_initialize_sets_independent_optimizer_copies(self):
        self.assertIsNotNone(self.layer.w_opt)
        self.assertIsNotNone(self.layer.b_opt)
        self.assertIsNot(self.layer.w_opt, self.layer.b_opt)

    def test_backward_propagation_shapes(self):
        out = self.layer.forward_propagation(self.X, training=True)
        output_error = np.ones_like(out)
        input_error = self.layer.backward_propagation(output_error)
        self.assertEqual(input_error.shape, self.X.shape)

    def test_backward_propagation_gradient_check(self):
        np.random.seed(0)
        X = np.random.rand(4, 3)
        Y = np.random.rand(4, 2)

        mse = lambda y, p: np.mean((y - p) ** 2)
        mse_derivative = lambda y, p: 2 * (p - y) / y.size

        layer = DenseLayer(2, (3,)).initialize(NoUpdate())
        out = layer.forward_propagation(X, training=True)
        analytic = X.T @ mse_derivative(Y, out)

        numeric, eps = np.zeros_like(layer.weights), 1e-6
        for i in range(layer.weights.shape[0]):
            for j in range(layer.weights.shape[1]):
                layer.weights[i, j] += eps
                lp = mse(Y, layer.forward_propagation(X, True))
                layer.weights[i, j] -= 2 * eps
                lm = mse(Y, layer.forward_propagation(X, True))
                layer.weights[i, j] += eps
                numeric[i, j] = (lp - lm) / (2 * eps)

        self.assertLess(np.abs(numeric - analytic).max(), 1e-8)

    def test_layer_name(self):
        self.assertEqual(self.layer.layer_name(), "DenseLayer")


class TestDropout(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.X = np.random.rand(1000, 5)
        self.layer = Dropout(probability=0.5)
        self.layer.set_input_shape((5,))

    def test_training_mode_zeroes_some_values(self):
        out = self.layer.forward_propagation(self.X, training=True)
        self.assertTrue(np.any(out == 0))

    def test_training_mode_scales_nonzero_values(self):
        out = self.layer.forward_propagation(self.X, training=True)
        # media aproximada mantida graças ao scaling (lei dos grandes numeros)
        self.assertAlmostEqual(out.mean(), self.X.mean(), delta=0.05)

    def test_inference_mode_returns_input_unchanged(self):
        out = self.layer.forward_propagation(self.X, training=False)
        np.testing.assert_array_equal(out, self.X)

    def test_output_shape(self):
        self.assertEqual(self.layer.output_shape(), (5,))

    def test_parameters(self):
        self.assertEqual(self.layer.parameters(), 0)

    def test_backward_propagation_uses_mask(self):
        out = self.layer.forward_propagation(self.X, training=True)
        output_error = np.ones_like(self.X)
        input_error = self.layer.backward_propagation(output_error)
        np.testing.assert_array_equal(input_error, self.layer.mask)


if __name__ == "__main__":
    unittest.main()
