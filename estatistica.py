"""Funções estatísticas do trabalho: tamanho amostral, intervalos de confiança e testes.

Todas as funções recebem dados ou resumos da amostra e devolvem dicionários simples,
para que o painel exiba os mesmos números que o README documenta.
"""

import math

import numpy as np
from scipy import stats


# ===========================================================================
# Tamanho amostral
# ===========================================================================

def tamanho_amostral_proporcao(N, confianca=0.95, erro=0.05, p=0.5):
    """Tamanho da amostra para estimar uma proporção com correção para população finita.

    É a mesma fórmula da calculadora da USP/Bauru:
        n0 = z² p (1 - p) / e²
        n  = n0 / (1 + (n0 - 1) / N)
    """
    z = stats.norm.ppf(1 - (1 - confianca) / 2)
    n0 = z**2 * p * (1 - p) / erro**2
    n = n0 / (1 + (n0 - 1) / N)
    return {"z": z, "n0": n0, "n_exato": n, "n": math.ceil(n)}


# ===========================================================================
# Proporção
# ===========================================================================

def inferencia_proporcao(x, n, p0, alternativa, alfa=0.05, confianca=0.95):
    """IC de Wald e teste Z para uma proporção.

    alternativa: "maior" (H1: p > p0), "menor" (H1: p < p0) ou "diferente".
    """
    p_hat = x / n
    z_crit = stats.norm.ppf(1 - (1 - confianca) / 2)
    ep_ic = math.sqrt(p_hat * (1 - p_hat) / n)
    ic = (p_hat - z_crit * ep_ic, p_hat + z_crit * ep_ic)

    # No teste o erro-padrão usa p0 (vale sob H0)
    ep_teste = math.sqrt(p0 * (1 - p0) / n)
    z = (p_hat - p0) / ep_teste
    p_valor = calcular_p_valor(stats.norm, z, alternativa)

    return {
        "n": n,
        "x": x,
        "estimativa": p_hat,
        "erro_padrao": ep_ic,
        "margem": z_crit * ep_ic,
        "ic": ic,
        "estatistica": z,
        "nome_estatistica": "Z",
        "p_valor": p_valor,
        "rejeita_h0": p_valor < alfa,
        "critico": valores_criticos(stats.norm, alternativa, alfa),
        "distribuicao": stats.norm,
        "condicoes": {
            "n·p̂ ≥ 10": (n * p_hat, n * p_hat >= 10),
            "n·(1 − p̂) ≥ 10": (n * (1 - p_hat), n * (1 - p_hat) >= 10),
            "n·p₀ ≥ 10": (n * p0, n * p0 >= 10),
            "n·(1 − p₀) ≥ 10": (n * (1 - p0), n * (1 - p0) >= 10),
        },
    }


# ===========================================================================
# Média
# ===========================================================================

def inferencia_media(valores, mu0, alternativa, alfa=0.05, confianca=0.95):
    """IC t de Student e teste t para uma média (σ desconhecido)."""
    valores = np.asarray(valores, dtype=float)
    n = len(valores)
    media = valores.mean()
    s = valores.std(ddof=1)
    gl = n - 1
    dist = stats.t(gl)

    ep = s / math.sqrt(n)
    t_crit = dist.ppf(1 - (1 - confianca) / 2)
    ic = (media - t_crit * ep, media + t_crit * ep)

    t = (media - mu0) / ep
    p_valor = calcular_p_valor(dist, t, alternativa)
    assimetria = stats.skew(valores)

    return {
        "n": n,
        "estimativa": media,
        "desvio_padrao": s,
        "erro_padrao": ep,
        "margem": t_crit * ep,
        "ic": ic,
        "gl": gl,
        "estatistica": t,
        "nome_estatistica": "t",
        "p_valor": p_valor,
        "rejeita_h0": p_valor < alfa,
        "critico": valores_criticos(dist, alternativa, alfa),
        "distribuicao": dist,
        "assimetria": assimetria,
        "condicoes": {
            # Regra prática para distribuições muito assimétricas: n > 15·|assimetria|²
            "n ≥ 30 (TCL)": (n, n >= 30),
            "n > 15·assimetria²": (15 * assimetria**2, n > 15 * assimetria**2),
        },
    }


def ic_bootstrap_media(valores, confianca=0.95, reamostras=5000, semente=2026):
    """IC bootstrap percentil para a média, usado como verificação do IC t."""
    valores = np.asarray(valores, dtype=float)
    rng = np.random.default_rng(semente)
    medias = rng.choice(valores, size=(reamostras, len(valores)), replace=True).mean(axis=1)
    alfa = 1 - confianca
    return tuple(np.quantile(medias, [alfa / 2, 1 - alfa / 2]))


# ===========================================================================
# Auxiliares
# ===========================================================================

def calcular_p_valor(dist, estatistica, alternativa):
    if alternativa == "maior":
        return dist.sf(estatistica)
    if alternativa == "menor":
        return dist.cdf(estatistica)
    return 2 * dist.sf(abs(estatistica))


def valores_criticos(dist, alternativa, alfa):
    """Valor(es) crítico(s) da estatística de teste que delimitam a região de rejeição."""
    if alternativa == "maior":
        return (dist.ppf(1 - alfa),)
    if alternativa == "menor":
        return (dist.ppf(alfa),)
    return (dist.ppf(alfa / 2), dist.ppf(1 - alfa / 2))


def descritivas(valores):
    """Medidas descritivas de uma variável quantitativa."""
    valores = np.asarray(valores, dtype=float)
    q1, mediana, q3 = np.quantile(valores, [0.25, 0.5, 0.75])
    media = valores.mean()
    dp = valores.std(ddof=1)
    return {
        "n": len(valores),
        "Média": media,
        "Mediana": mediana,
        "Desvio-padrão": dp,
        "Coef. de variação": dp / media,
        "Mínimo": valores.min(),
        "1º quartil": q1,
        "3º quartil": q3,
        "Máximo": valores.max(),
        "Amplitude interquartil": q3 - q1,
        "Assimetria": stats.skew(valores),
        "Curtose (excesso)": stats.kurtosis(valores),
    }
