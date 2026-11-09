import copy
import itertools

import numpy as np

from si.data.dataset import Dataset


def _make_folds(n_samples, cv, seed=None):
    """
    Divide os índices das amostras em cv folds (criados UMA vez, para que todas
    as configurações sejam comparadas exatamente nos mesmos folds).

    Returns
    -------
    list[np.ndarray]
        Lista com os índices de validação de cada fold.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return np.array_split(indices, cv)


def _subset(dataset, indices):
    """Cria um novo Dataset só com as amostras indicadas."""
    return Dataset(dataset.X[indices],
                   dataset.y[indices] if dataset.y is not None else None,
                   features=dataset.features,
                   label=dataset.label)


def _new_model(model, configuration):
    """
    Cria um modelo NOVO (sem treino) com a configuração dada.

    - Se model for uma função (builder), chama-a com a configuração: model(**configuration).
    - Se for um modelo, faz uma cópia profunda e muda-lhe os atributos (setattr).
    """
    if callable(model) and not hasattr(model, 'fit'):
        return model(**configuration)
    new_model = copy.deepcopy(model)
    for name, value in configuration.items():
        setattr(new_model, name, value)
    return new_model


def _cross_validate(model, dataset, configuration, folds, scoring=None):
    """
    Validação cruzada de uma configuração: para cada fold constrói um modelo novo,
    treina nos folds de treino e avalia no fold de validação.

    Returns
    -------
    float
        Média dos scores de validação dos k folds.
    """
    n_samples = dataset.X.shape[0]
    scores = []
    for i, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        train = _subset(dataset, train_idx)
        validation = _subset(dataset, val_idx)

        fold_model = _new_model(model, configuration)
        fold_model.fit(train)

        if scoring is None:
            score = fold_model.score(validation)
        else:
            predictions = fold_model.predict(validation)
            # saída (n, 1) com y 1D: achatar para (n,) para não haver broadcasting
            if validation.y.ndim == 1 and np.ndim(predictions) == 2 and predictions.shape[1] == 1:
                predictions = predictions.ravel()
            score = scoring(validation.y, predictions)
        scores.append(score)
    return float(np.mean(scores))


def _results(configurations, scores):
    """Monta o dicionário de resultados (a configuração com maior score é a melhor)."""
    best = int(np.argmax(scores))  # em caso de empate, fica a primeira
    return {
        'hyperparameters': configurations,
        'scores': scores,
        'best_hyperparameters': configurations[best],
        'best_score': scores[best],
    }


def grid_search_cv(model, dataset, hyperparameter_grid, scoring=None, cv=5, seed=None):
    """
    Grid search com validação cruzada: testa TODAS as combinações de hiperparâmetros.

    Parameters
    ----------
    model : Model ou callable
        O modelo a afinar, ou uma função (builder) que devolve um modelo novo
        quando chamada com os hiperparâmetros: model(**configuração).
    dataset : Dataset
        Dataset de treino (nunca o de teste).
    hyperparameter_grid : dict
        {nome: lista de valores a experimentar}.
    scoring : callable, opcional
        Função scoring(y_true, y_pred). Se None, usa model.score.
    cv : int
        Número de folds.
    seed : int, opcional
        Semente usada para criar os folds.

    Returns
    -------
    dict
        'hyperparameters' (todas as configurações testadas), 'scores' (score médio de
        cada uma), 'best_hyperparameters' e 'best_score'.
    """
    # 1. criar os folds uma só vez (os mesmos para todas as configurações)
    folds = _make_folds(dataset.X.shape[0], cv, seed)

    # 2. todas as combinações de valores
    combinations = itertools.product(*hyperparameter_grid.values())

    configurations, scores = [], []
    # 3. para cada combinação, validação cruzada com modelos novos
    for combination in combinations:
        configuration = dict(zip(hyperparameter_grid.keys(), combination))
        score = _cross_validate(model, dataset, configuration, folds, scoring)
        configurations.append(configuration)
        scores.append(score)

    # 4. e 5. escolher a melhor e devolver
    return _results(configurations, scores)
