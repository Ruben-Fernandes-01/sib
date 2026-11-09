import numpy as np

from si.model_selection.grid_search import _make_folds, _cross_validate, _results


def log_uniform(low, high):
    """
    Devolve uma função que amostra numa escala logarítmica entre low e high:
    10 ** uniform(log10(low), log10(high)).

    A função devolvida recebe o gerador aleatório (np.random.Generator).
    """
    def sampler(rng):
        return 10 ** rng.uniform(np.log10(low), np.log10(high))
    return sampler


def randomized_search_cv(model, dataset, hyperparameter_distributions, scoring=None,
                         cv=5, n_iter=10, seed=None):
    """
    Pesquisa aleatória com validação cruzada: testa n_iter configurações sorteadas.

    Parameters
    ----------
    model : Model ou callable
        O modelo a afinar, ou uma função (builder) que devolve um modelo novo.
    dataset : Dataset
        Dataset de treino (nunca o de teste).
    hyperparameter_distributions : dict
        {nome: lista de valores (escolha uniforme) ou função amostradora (sampler)}.
        Um sampler é chamado com o gerador aleatório: sampler(rng).
    scoring : callable, opcional
        Função scoring(y_true, y_pred). Se None, usa model.score.
    cv : int
        Número de folds.
    n_iter : int
        Número de configurações aleatórias a avaliar.
    seed : int, opcional
        Semente usada para criar os folds e o gerador aleatório.

    Returns
    -------
    dict
        O mesmo dicionário do grid_search_cv.
    """
    # folds criados uma só vez e gerador aleatório com a semente
    folds = _make_folds(dataset.X.shape[0], cv, seed)
    rng = np.random.default_rng(seed)

    configurations, scores = [], []
    for _ in range(n_iter):
        # sortear um valor para cada hiperparâmetro
        configuration = {}
        for name, distribution in hyperparameter_distributions.items():
            if callable(distribution):
                configuration[name] = distribution(rng)
            else:
                configuration[name] = distribution[rng.integers(len(distribution))]

        score = _cross_validate(model, dataset, configuration, folds, scoring)
        configurations.append(configuration)
        scores.append(score)

    return _results(configurations, scores)
