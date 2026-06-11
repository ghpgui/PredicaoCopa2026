import numpy as np
from scipy.stats import poisson as scipy_poisson

MAX_GOLS = 10


def probabilidades_resultado(lambda_casa, lambda_visit, max_gols=MAX_GOLS):
    """Retorna (p_vitoria, p_empate, p_derrota) via grade de dois Poisson independentes."""
    gols = np.arange(max_gols + 1)
    p_c = scipy_poisson.pmf(gols, lambda_casa)
    p_v = scipy_poisson.pmf(gols, lambda_visit)
    grade = np.outer(p_c, p_v)  # grade[i,j] = P(home=i, away=j)

    idx_vitoria = np.tril_indices(max_gols + 1, k=-1)  # i > j → home vence
    p_vitoria = float(grade[idx_vitoria].sum())
    p_empate = float(np.trace(grade))
    p_derrota = 1.0 - p_vitoria - p_empate
    return p_vitoria, p_empate, p_derrota


def resultado_previsto(lambda_casa, lambda_visit):
    """Retorna 'V', 'E' ou 'D' pelo argmax das probabilidades Poisson."""
    pv, pe, pd = probabilidades_resultado(lambda_casa, lambda_visit)
    idx = np.argmax([pv, pe, pd])
    return ["V", "E", "D"][idx]


def resultado_real(gols_casa, gols_visitante):
    """Retorna 'V', 'E' ou 'D' a partir dos gols reais."""
    if gols_casa > gols_visitante:
        return "V"
    if gols_casa == gols_visitante:
        return "E"
    return "D"
