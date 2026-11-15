"""Prepara a população e sorteia a amostra aleatória simples (AAS).

Uso:
    python preparar_dados.py [caminho/para/Ano-2025.csv]

Gera:
    dados/populacao.parquet  - população completa (somente colunas usadas)
    dados/amostra.csv        - AAS sorteada sem reposição, com semente fixa
    dados/metadados.json     - contagens do recorte e parâmetros do dimensionamento
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from estatistica import tamanho_amostral_proporcao

PASTA = Path(__file__).parent
CAMINHO_ORIGEM = Path(r"C:\UFRGS\IA001\Real\Ano-2025.csv")
CAMINHO_POPULACAO = PASTA / "dados" / "populacao.parquet"
CAMINHO_AMOSTRA = PASTA / "dados" / "amostra.csv"
CAMINHO_METADADOS = PASTA / "dados" / "metadados.json"

# Parâmetros do dimensionamento (justificados no README e no painel)
CONFIANCA = 0.95
ERRO = 0.05
P_PLANEJAMENTO = 0.5
# Semente do gerador: torna o sorteio reproduzível por qualquer pessoa
SEMENTE = 2026

COLUNAS = [
    "ideDocumento", "nuDeputadoId", "txNomeParlamentar", "sgPartido", "sgUF",
    "txtDescricao", "datEmissao", "numMes", "vlrLiquido",
]

# Nomes curtos das categorias de despesa, para caberem nos gráficos
CATEGORIAS_CURTAS = {
    "COMBUSTÍVEIS E LUBRIFICANTES.": "Combustíveis",
    "PASSAGEM AÉREA - SIGEPA": "Passagem aérea",
    "PASSAGEM AÉREA - RPA": "Passagem aérea",
    "PASSAGEM AÉREA - REEMBOLSO": "Passagem aérea",
    "MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE PARLAMENTAR": "Escritório",
    "SERVIÇO DE TÁXI, PEDÁGIO E ESTACIONAMENTO": "Táxi, pedágio e estac.",
    "DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR.": "Divulgação",
    "TELEFONIA": "Telefonia",
    "FORNECIMENTO DE ALIMENTAÇÃO DO PARLAMENTAR": "Alimentação",
    "HOSPEDAGEM ,EXCETO DO PARLAMENTAR NO DISTRITO FEDERAL.": "Hospedagem",
    "LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES": "Locação de veículos",
}
OUTRAS = "Outras"

REGIOES = {
    "Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MS", "MT"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}
UF_REGIAO = {uf: regiao for regiao, ufs in REGIOES.items() for uf in ufs}


def carregar_populacao(caminho):
    df = pd.read_csv(caminho, sep=";", encoding="utf-8-sig", low_memory=False)
    recorte = {"registros_originais": len(df), "variaveis_originais": df.shape[1]}
    df = df[COLUNAS]

    # Mesmo recorte da IA001: lançamentos de deputados com partido e valor positivo
    # (lideranças não têm partido; valores negativos são estornos)
    com_partido = df["sgPartido"].notna() & (df["sgPartido"] != "S.PART.")
    recorte["removidos_sem_partido"] = int((~com_partido).sum())
    df = df[com_partido]
    recorte["removidos_valor_nao_positivo"] = int((df["vlrLiquido"] <= 0).sum())
    df = df[df["vlrLiquido"] > 0]

    df["categoria"] = df["txtDescricao"].map(CATEGORIAS_CURTAS).fillna(OUTRAS)
    df["regiao"] = df["sgUF"].map(UF_REGIAO)
    df["datEmissao"] = pd.to_datetime(df["datEmissao"], errors="coerce")
    df["numMes"] = df["numMes"].astype(int)

    df = df.reset_index(drop=True)
    # Identificador da unidade amostral (um lançamento = uma linha)
    df.insert(0, "id_populacao", np.arange(len(df)))
    return df, recorte


def sortear_aas(df, n, semente):
    """AAS sem reposição: cada lançamento tem a mesma probabilidade n/N de ser escolhido."""
    rng = np.random.default_rng(semente)
    indices = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(indices)]


def main():
    origem = Path(sys.argv[1]) if len(sys.argv) > 1 else CAMINHO_ORIGEM
    populacao, recorte = carregar_populacao(origem)
    N = len(populacao)

    calculo = tamanho_amostral_proporcao(N, CONFIANCA, ERRO, P_PLANEJAMENTO)
    amostra = sortear_aas(populacao, calculo["n"], SEMENTE)

    CAMINHO_POPULACAO.parent.mkdir(exist_ok=True)
    populacao.to_parquet(CAMINHO_POPULACAO, index=False, compression="zstd")
    amostra.to_csv(CAMINHO_AMOSTRA, index=False, encoding="utf-8-sig")

    metadados = {
        **recorte,
        "N": N,
        "confianca": CONFIANCA,
        "erro": ERRO,
        "p_planejamento": P_PLANEJAMENTO,
        "z": calculo["z"],
        "n0": calculo["n0"],
        "n_exato": calculo["n_exato"],
        "n": calculo["n"],
        "semente": SEMENTE,
    }
    CAMINHO_METADADOS.write_text(json.dumps(metadados, indent=2), encoding="utf-8")

    print(f"Registros no arquivo original: {recorte['registros_originais']:,}")
    print(f"População (após recorte):      {N:,}")
    print(f"z = {calculo['z']:.4f} | n0 = {calculo['n0']:.2f} | n corrigido = {calculo['n_exato']:.2f}")
    print(f"Tamanho da amostra:            {calculo['n']}")
    print(f"Semente:                       {SEMENTE}")
    print(f"Arquivos: {CAMINHO_POPULACAO.name} ({CAMINHO_POPULACAO.stat().st_size / 1e6:.1f} MB), "
          f"{CAMINHO_AMOSTRA.name}")


if __name__ == "__main__":
    main()
