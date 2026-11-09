import unittest

import numpy as np

from si.neural_networks.optimizers import Optimizer, SGD, Adam


class TestOptimizers(unittest.TestCase):

    def setUp(self):
        self.w = np.array([[1.0, 2.0]])
        self.g = np.array([[0.5, -0.5]])

    def test_base_class_is_abstract(self):
        with self.assertRaises(TypeError):
            Optimizer(0.1)

    def test_sgd_without_momentum(self):
        new_w = SGD(learning_rate=0.1).update(self.w, self.g)
        np.testing.assert_allclose(new_w, self.w - 0.1 * self.g)

    def test_sgd_initializes_retained_gradient(self):
        sgd = SGD(learning_rate=0.1, momentum=0.9)
        self.assertIsNone(sgd.retained_gradient)
        sgd.update(self.w, self.g)
        self.assertEqual(sgd.retained_gradient.shape, self.w.shape)

    def test_sgd_momentum_matches_manual_computation(self):
        sgd = SGD(learning_rate=0.1, momentum=0.9)
        w = self.w.copy()
        w_ref, v = self.w.copy(), np.zeros_like(self.w)
        for _ in range(3):
            w = sgd.update(w, self.g)
            v = 0.9 * v + 0.1 * self.g
            w_ref = w_ref - 0.1 * v
        np.testing.assert_allclose(w, w_ref)

    def test_adam_first_step_is_lr_times_sign(self):
        # no 1.º passo, m_hat = g e v_hat = g^2  ->  passo ≈ lr * sign(g)
        new_w = Adam(learning_rate=0.1).update(self.w, self.g)
        np.testing.assert_allclose(new_w, self.w - 0.1 * np.sign(self.g), atol=1e-6)

    def test_adam_updates_timestep_and_moments(self):
        adam = Adam(learning_rate=0.1)
        self.assertEqual(adam.t, 0)
        adam.update(self.w, self.g)
        adam.update(self.w, self.g)
        self.assertEqual(adam.t, 2)
        self.assertEqual(adam.m.shape, self.w.shape)
        self.assertEqual(adam.v.shape, self.w.shape)

    def test_adam_default_hyperparameters(self):
        adam = Adam(learning_rate=0.01)
        self.assertEqual((adam.beta_1, adam.beta_2, adam.epsilon), (0.9, 0.999, 1e-8))

    def test_adam_minimizes_quadratic(self):
        adam, x = Adam(learning_rate=0.1), np.array([5.0, -3.0])
        for _ in range(500):
            x = adam.update(x, 2 * x)
        self.assertLess(np.abs(x).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
