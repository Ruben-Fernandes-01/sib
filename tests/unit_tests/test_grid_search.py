import unittest

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.grid_search import grid_search_cv


class DummyModel:
    """Modelo de brincar: prevê sempre a constante 'c'; o score é a accuracy."""

    def __init__(self, c=0.0, noise=0):
        self.c = c
        self.noise = noise
        self.fitted = False

    def fit(self, dataset):
        self.fitted = True
        return self

    def predict(self, dataset):
        return np.full(dataset.X.shape[0], self.c)

    def score(self, dataset):
        return float(np.mean(self.predict(dataset) == dataset.y))


class TestGridSearchCV(unittest.TestCase):

    def setUp(self):
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = np.array([1.0] * 15 + [0.0] * 5)    # 75% de classe 1
        self.ds = Dataset(X, y)

    def test_returns_expected_keys(self):
        r = grid_search_cv(DummyModel(), self.ds, {'c': [0.0, 1.0]}, cv=4, seed=0)
        self.assertEqual(set(r), {'hyperparameters', 'scores', 'best_hyperparameters', 'best_score'})

    def test_tests_all_combinations(self):
        grid = {'c': [0.0, 1.0], 'noise': [1, 2, 3]}
        r = grid_search_cv(DummyModel(), self.ds, grid, cv=4, seed=0)
        self.assertEqual(len(r['hyperparameters']), 6)
        self.assertEqual(len(r['scores']), 6)
        self.assertEqual(r['hyperparameters'][0], {'c': 0.0, 'noise': 1})
        self.assertEqual(r['hyperparameters'][1], {'c': 0.0, 'noise': 2})

    def test_finds_best_configuration(self):
        r = grid_search_cv(DummyModel(), self.ds, {'c': [0.0, 1.0]}, cv=4, seed=0)
        self.assertEqual(r['best_hyperparameters'], {'c': 1.0})
        self.assertAlmostEqual(r['best_score'], 0.75)

    def test_score_is_mean_of_folds(self):
        # com c=1 os scores dos folds são 5/5 e a média tem de ser 15/20 = 0.75
        r = grid_search_cv(DummyModel(), self.ds, {'c': [1.0]}, cv=5, seed=3)
        self.assertAlmostEqual(r['scores'][0], 0.75)

    def test_one_fresh_model_per_fold_and_configuration(self):
        base = DummyModel()
        grid_search_cv(base, self.ds, {'c': [0.0, 1.0]}, cv=4, seed=0)
        self.assertFalse(base.fitted)    # o modelo original nunca é treinado
        built = []

        def builder(c):
            m = DummyModel(c)
            built.append(m)
            return m

        grid_search_cv(builder, self.ds, {'c': [0.0, 1.0]}, cv=4, seed=0)
        self.assertEqual(len(built), 2 * 4)             # configurações x k
        self.assertEqual(len({id(m) for m in built}), 8)  # todos diferentes

    def test_same_folds_for_every_configuration(self):
        seen = []

        class Spy(DummyModel):
            def fit(self, dataset):
                seen.append(tuple(dataset.X[:, 0]))
                return super().fit(dataset)

        grid_search_cv(lambda c: Spy(c), self.ds, {'c': [0.0, 1.0]}, cv=4, seed=7)
        self.assertEqual(seen[:4], seen[4:])

    def test_custom_scoring_function(self):
        scoring = lambda y_true, y_pred: 42.0
        r = grid_search_cv(DummyModel(), self.ds, {'c': [0.0, 1.0]}, scoring=scoring, cv=4, seed=0)
        self.assertEqual(r['scores'], [42.0, 42.0])

    def test_ties_pick_first_configuration(self):
        scoring = lambda y_true, y_pred: 1.0
        r = grid_search_cv(DummyModel(), self.ds, {'c': [0.0, 1.0]}, scoring=scoring, cv=4, seed=0)
        self.assertEqual(r['best_hyperparameters'], {'c': 0.0})

    def test_model_instance_is_copied_and_attributes_set(self):
        base = DummyModel(c=0.0)
        grid_search_cv(base, self.ds, {'c': [1.0]}, cv=4, seed=0)
        self.assertEqual(base.c, 0.0)    # o original não foi alterado

    def test_reproducible_with_seed(self):
        a = grid_search_cv(DummyModel(), self.ds, {'c': [0.0, 1.0]}, cv=4, seed=5)
        b = grid_search_cv(DummyModel(), self.ds, {'c': [0.0, 1.0]}, cv=4, seed=5)
        self.assertEqual(a['scores'], b['scores'])


if __name__ == "__main__":
    unittest.main()
