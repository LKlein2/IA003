"""Dados, formatação, gráficos e definições compartilhados pelas páginas do painel."""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from estatistica import (
    calcular_p_valor,
    descritivas,
    inferencia_media,
    inferencia_proporcao,
)

# ===========================================================================
# Caminhos e cores
# ===========================================================================

PASTA = Path(__file__).parent
CAMINHO_POPULACAO = PASTA / "dados" / "populacao.parquet"
CAMINHO_AMOSTRA = PASTA / "dados" / "amostra.csv"
CAMINHO_METADADOS = PASTA / "dados" / "metadados.json"

POPULACAO = "População"
AMOSTRA = "Amostra"
# População em azul e amostra em laranja em todos os gráficos (slots 1 e 2 da paleta)
CORES_CONJUNTO = {POPULACAO: "#2a78d6", AMOSTRA: "#eb6834"}
COR_H0 = "#7a7974"
COR_REJEICAO = "#d03b3b"
COR_ACEITACAO = "#a3a29c"

FONTE = (
    "Fonte: Câmara dos Deputados - Dados Abertos, Cota para o Exercício da Atividade "
    "Parlamentar (CEAP), arquivo Ano-2025.csv."
)

# ===========================================================================
# Variáveis analisadas
# ===========================================================================

# Variáveis quantitativas: rótulo -> filtro de linhas (None = todas as despesas)
VARIAVEIS_QUANTITATIVAS = {
    "Valor líquido da despesa (R$)": None,
    "Valor da passagem aérea (R$)": "Passagem aérea",
}

# Variáveis qualitativas: rótulo -> coluna
VARIAVEIS_QUALITATIVAS = {
    "Categoria de despesa": "categoria",
    "Região do parlamentar": "regiao",
}

REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


# ===========================================================================
# Carga dos dados
# ===========================================================================

@st.cache_data
def carregar_populacao():
    return pd.read_parquet(CAMINHO_POPULACAO)


@st.cache_data
def carregar_amostra():
    return pd.read_csv(CAMINHO_AMOSTRA, encoding="utf-8-sig", parse_dates=["datEmissao"])


@st.cache_data
def carregar_metadados():
    return json.loads(CAMINHO_METADADOS.read_text(encoding="utf-8"))


def valores_quantitativos(df, variavel):
    categoria = VARIAVEIS_QUANTITATIVAS[variavel]
    if categoria is None:
        return df["vlrLiquido"]
    return df.loc[df["categoria"] == categoria, "vlrLiquido"]


# ===========================================================================
# Formatação em pt-BR
# ===========================================================================

def fmt_num(valor, casas=0):
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_reais(valor, casas=2):
    sinal = "−" if valor < 0 else ""
    return f"{sinal}R$ {fmt_num(abs(valor), casas)}"


def fmt_pct(valor, casas=1):
    return f"{fmt_num(valor * 100, casas)}%"


def fmt_milhoes(valor):
    return f"R$ {fmt_num(valor / 1e6, 1)} mi"


def fmt_p_valor(valor):
    return "< 0,0001" if valor < 0.0001 else fmt_num(valor, 4)


def fmt_variavel(valor, eh_proporcao, casas=None):
    if eh_proporcao:
        return fmt_pct(valor, 1 if casas is None else casas)
    return fmt_reais(valor, 2 if casas is None else casas)


# Rótulos de eixo em pt-BR (o Vega-Lite formata em inglês)
EIXO_REAIS = "replace(format(datum.value, ',.0f'), regexp(',', 'g'), '.')"
EIXO_PCT = "replace(format(datum.value, '.0%'), '.', ',')"
EIXO_DECIMAL = "replace(format(datum.value, '.1f'), '.', ',')"


# ===========================================================================
# Tabelas de frequência
# ===========================================================================

def ordem_niveis(populacao, coluna):
    """Ordem fixa dos níveis, definida pela população, para que amostra e população coincidam."""
    if coluna == "regiao":
        return REGIOES
    return list(populacao[coluna].value_counts().index)


def tabela_frequencias(df, coluna, ordem):
    freq = df[coluna].value_counts().reindex(ordem, fill_value=0)
    return pd.DataFrame({"Nível": ordem, "Frequência": freq.to_numpy(), "Proporção": (freq / len(df)).to_numpy()})


# ===========================================================================
# Gráficos descritivos
# ===========================================================================

def bins_quantitativos(valores_referencia, escala):
    """Limites dos intervalos do histograma, calculados na população para valerem para os dois conjuntos."""
    if escala == "Logarítmica":
        return np.logspace(0, np.log10(valores_referencia.max()), 41)
    return np.linspace(0, np.quantile(valores_referencia, 0.99), 41)


def contar_bins(valores, bins, escala):
    # Na escala log, valores abaixo de R$ 1 entram no primeiro intervalo
    if escala == "Logarítmica":
        valores = valores.clip(lower=bins[0])
    contagem, _ = np.histogram(valores, bins=bins)
    return pd.DataFrame({
        "inicio": bins[:-1],
        "fim": bins[1:],
        "Lançamentos": contagem,
        "Proporção": contagem / len(valores),
        "Intervalo": [f"{fmt_reais(a)} a {fmt_reais(b)}" for a, b in zip(bins[:-1], bins[1:])],
    })


def escala_valores(escala, maximo):
    """Escala e eixo de valores em R$; na log, marcas em potências de 10 a partir de R$ 1."""
    if escala == "Logarítmica":
        marcas = [10**k for k in range(0, int(np.ceil(np.log10(maximo))) + 1)]
        return alt.Scale(type="log", domain=[1, marcas[-1]], clamp=True), alt.Axis(labelExpr=EIXO_REAIS, values=marcas)
    return alt.Scale(zero=True), alt.Axis(labelExpr=EIXO_REAIS)


def grafico_histograma(valores, conjunto, escala, bins, referencia=None):
    """Histograma em proporção. Com referência, a população entra como linha em degraus."""
    dados = contar_bins(valores, bins, escala)
    tipo_escala, eixo = escala_valores(escala, bins[-1])
    eixo_x = alt.X("inicio:Q", scale=tipo_escala, title="Valor (R$)", axis=eixo)
    tooltip = [
        alt.Tooltip("Intervalo:N"),
        alt.Tooltip("Lançamentos:Q", format=",d"),
        alt.Tooltip("Proporção:Q", format=".2%"),
    ]
    # mark_rect com y2 = 0: com x e x2 contínuos, mark_bar desenharia barras horizontais
    barras = alt.Chart(dados).mark_rect(color=CORES_CONJUNTO[conjunto], stroke="white", strokeWidth=1).encode(
        x=eixo_x,
        x2="fim:Q",
        y=alt.Y("Proporção:Q", title="% dos lançamentos", axis=alt.Axis(labelExpr=EIXO_PCT)),
        y2=alt.datum(0),
        tooltip=tooltip,
    )
    if referencia is None:
        return barras.properties(height=300)

    dados_ref = contar_bins(referencia, bins, escala)
    # Repete o último degrau para fechar a linha no fim do último intervalo
    fechamento = dados_ref.iloc[[-1]].assign(inicio=dados_ref["fim"].iloc[-1])
    dados_ref = pd.concat([dados_ref, fechamento], ignore_index=True)
    linha = alt.Chart(dados_ref).mark_line(interpolate="step-after", color=CORES_CONJUNTO[POPULACAO], strokeWidth=2).encode(
        x=eixo_x,
        y="Proporção:Q",
        tooltip=tooltip,
    )
    return (barras + linha).properties(height=300)


def resumo_boxplot(valores, conjunto):
    q1, mediana, q3 = np.quantile(valores, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    # Bigodes de Tukey: até o dado mais extremo dentro de 1,5·AIQ
    bigode_inf = valores[valores >= q1 - 1.5 * iqr].min()
    bigode_sup = valores[valores <= q3 + 1.5 * iqr].max()
    return {
        "Conjunto": conjunto,
        "Q1": q1, "Mediana": mediana, "Q3": q3, "Média": valores.mean(),
        "Bigode inferior": bigode_inf, "Bigode superior": bigode_sup,
        "Atípicos": (valores > bigode_sup).mean() + (valores < bigode_inf).mean(),
    }


def grafico_boxplot(conjuntos, escala):
    """Boxplot a partir de quartis já calculados (evita mandar 200 mil pontos ao navegador)."""
    resumo = pd.DataFrame([resumo_boxplot(v.to_numpy(), nome) for nome, v in conjuntos.items()])
    for coluna in ["Q1", "Mediana", "Q3", "Média", "Bigode inferior", "Bigode superior"]:
        resumo[f"{coluna} (texto)"] = resumo[coluna].map(fmt_reais)
    resumo["Atípicos (texto)"] = resumo["Atípicos"].map(fmt_pct)

    tipo_escala, eixo = escala_valores(escala, resumo["Bigode superior"].max())
    cor = alt.Color("Conjunto:N", scale=alt.Scale(domain=list(CORES_CONJUNTO), range=list(CORES_CONJUNTO.values())), legend=None)
    y = alt.Y("Conjunto:N", title=None, sort=list(CORES_CONJUNTO))
    tooltip = [alt.Tooltip(f"{c} (texto):N", title=c) for c in
               ["Bigode inferior", "Q1", "Mediana", "Média", "Q3", "Bigode superior", "Atípicos"]]

    base = alt.Chart(resumo).encode(y=y, tooltip=tooltip)
    bigodes = base.mark_rule(strokeWidth=1.5).encode(
        x=alt.X("Bigode inferior:Q", scale=tipo_escala, title="Valor (R$)", axis=eixo), x2="Bigode superior:Q", color=cor
    )
    caixa = base.mark_bar(size=26, opacity=0.35, cornerRadius=4).encode(x="Q1:Q", x2="Q3:Q", color=cor)
    mediana = base.mark_tick(size=26, thickness=3).encode(x="Mediana:Q", color=cor)
    media = base.mark_point(shape="diamond", size=90, filled=True, color="#52514e").encode(x="Média:Q")
    altura = 70 * len(resumo) + 40
    return (bigodes + caixa + mediana + media).properties(height=altura), resumo


def grafico_frequencias(tabela, conjunto):
    dados = tabela.assign(Rótulo=tabela["Proporção"].map(fmt_pct))
    y = alt.Y("Nível:N", sort=list(tabela["Nível"]), title=None, axis=alt.Axis(labelLimit=200))
    base = alt.Chart(dados).encode(
        y=y,
        x=alt.X("Proporção:Q", title="% dos lançamentos", axis=alt.Axis(labelExpr=EIXO_PCT)),
        tooltip=[alt.Tooltip("Nível:N"), alt.Tooltip("Frequência:Q", format=",d"), alt.Tooltip("Proporção:Q", format=".2%")],
    )
    barras = base.mark_bar(color=CORES_CONJUNTO[conjunto], cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
    rotulos = base.mark_text(align="left", dx=4, color="#52514e").encode(text="Rótulo:N")
    return (barras + rotulos).properties(height=max(180, 32 * len(dados)))


def grafico_frequencias_comparadas(tab_amostra, tab_populacao):
    """Gráfico de halteres: proporção de cada nível na amostra e na população."""
    dados = pd.concat([
        tab_populacao.assign(Conjunto=POPULACAO),
        tab_amostra.assign(Conjunto=AMOSTRA),
    ])
    dados["Proporção (texto)"] = dados["Proporção"].map(lambda v: fmt_pct(v, 2))
    y = alt.Y("Nível:N", sort=list(tab_populacao["Nível"]), title=None, axis=alt.Axis(labelLimit=200))
    x = alt.X("Proporção:Q", title="% dos lançamentos", axis=alt.Axis(labelExpr=EIXO_PCT))
    cor = alt.Color(
        "Conjunto:N",
        scale=alt.Scale(domain=list(CORES_CONJUNTO), range=list(CORES_CONJUNTO.values())),
        legend=alt.Legend(orient="top", title=None),
    )
    hastes = alt.Chart(dados).mark_line(color=COR_ACEITACAO, strokeWidth=2).encode(y=y, x=x, detail="Nível:N")
    pontos = alt.Chart(dados).mark_circle(size=110, opacity=1, stroke="white", strokeWidth=1.5).encode(
        y=y, x=x, color=cor,
        tooltip=[alt.Tooltip("Nível:N"), alt.Tooltip("Conjunto:N"), alt.Tooltip("Frequência:Q", format=",d"),
                 alt.Tooltip("Proporção (texto):N", title="Proporção")],
    )
    return (hastes + pontos).properties(height=max(180, 32 * len(tab_populacao)))


# ===========================================================================
# Painel descritivo (usado pela população e pela amostra)
# ===========================================================================

def filtrar(df, regioes):
    return df[df["regiao"].isin(regioes)]


def painel_descritivo(df, conjunto, referencia=None):
    """Monta o painel descritivo de `df`. Com `referencia` (população), compara os dois conjuntos."""
    comparando = referencia is not None
    prefixo = "amostra" if comparando else "populacao"

    # ------------------------------------------------------------------
    # Indicadores
    # ------------------------------------------------------------------
    st.subheader("Indicadores gerais", divider="gray")
    indicadores = [
        ("Lançamentos", len(df), fmt_num, None),
        ("Deputados com despesa", df["nuDeputadoId"].nunique(), fmt_num, None),
        ("Valor total", df["vlrLiquido"].sum(), fmt_milhoes, None),
        ("Média por lançamento", df["vlrLiquido"].mean(), fmt_reais, "media"),
        ("Mediana por lançamento", df["vlrLiquido"].median(), fmt_reais, "mediana"),
        ("Lançamentos do Sudeste", (df["regiao"] == "Sudeste").mean(), fmt_pct, "sudeste"),
    ]
    with st.container(horizontal=True):
        for rotulo, valor, formato, chave in indicadores:
            delta = None
            if comparando:
                if chave == "media":
                    delta = f"{formato(referencia['vlrLiquido'].mean())} na população"
                elif chave == "mediana":
                    delta = f"{formato(referencia['vlrLiquido'].median())} na população"
                elif chave == "sudeste":
                    delta = f"{formato((referencia['regiao'] == 'Sudeste').mean())} na população"
                elif rotulo == "Lançamentos":
                    delta = f"{fmt_pct(len(df) / len(referencia), 2)} da população"
                elif rotulo == "Deputados com despesa":
                    delta = f"de {fmt_num(referencia['nuDeputadoId'].nunique())} na população"
                else:
                    delta = f"{fmt_pct(df['vlrLiquido'].sum() / referencia['vlrLiquido'].sum(), 2)} do total"
            st.metric(rotulo, formato(valor), delta=delta, delta_color="off", delta_arrow="off", border=True)

    # ------------------------------------------------------------------
    # Variáveis quantitativas
    # ------------------------------------------------------------------
    st.subheader("Variáveis quantitativas", divider="gray")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        variavel = st.segmented_control(
            "Variável", list(VARIAVEIS_QUANTITATIVAS), default=list(VARIAVEIS_QUANTITATIVAS)[0],
            required=True, key=f"{prefixo}_quanti",
        )
        escala = st.segmented_control(
            "Escala do eixo de valores", ["Logarítmica", "Linear"], default="Logarítmica", required=True,
            key=f"{prefixo}_escala",
            help="Os valores são muito assimétricos. A escala linear mostra até o percentil 99 da população.",
        )

    valores = valores_quantitativos(df, variavel)
    valores_ref = valores_quantitativos(referencia, variavel) if comparando else None
    bins = bins_quantitativos(valores_ref if comparando else valores, escala)

    if valores.empty:
        st.warning("Nenhum lançamento desta variável na seleção.")
        return

    col_hist, col_box = st.columns([3, 2])
    with col_hist:
        titulo = "Distribuição (barras = amostra, linha = população)" if comparando else "Distribuição"
        st.markdown(f"**{titulo}**")
        st.altair_chart(
            grafico_histograma(valores, conjunto, escala, bins, valores_ref),
            alt=f"Histograma de {variavel.lower()}",
        )
        if escala == "Linear":
            acima = (valores > bins[-1]).mean()
            st.caption(f"{fmt_pct(acima, 2)} dos lançamentos ficam acima de {fmt_reais(bins[-1])} e não aparecem no gráfico.")
    with col_box:
        st.markdown("**Boxplot** (losango = média)")
        conjuntos = {POPULACAO: valores_ref, AMOSTRA: valores} if comparando else {conjunto: valores}
        grafico, resumo = grafico_boxplot(conjuntos, escala)
        st.altair_chart(grafico, alt=f"Boxplot de {variavel.lower()}")
        atipicos = ", ".join(f"{linha['Conjunto'].lower()} {fmt_pct(linha['Atípicos'])}" for _, linha in resumo.iterrows())
        st.caption(f"Valores atípicos (além de 1,5·AIQ) omitidos: {atipicos}.")

    st.markdown("**Medidas descritivas**")
    st.dataframe(tabela_descritivas(valores, valores_ref, conjunto), hide_index=True, alt="Medidas descritivas")

    # ------------------------------------------------------------------
    # Variáveis qualitativas
    # ------------------------------------------------------------------
    st.subheader("Variáveis qualitativas", divider="gray")
    variavel_quali = st.segmented_control(
        "Variável", list(VARIAVEIS_QUALITATIVAS), default=list(VARIAVEIS_QUALITATIVAS)[0],
        required=True, key=f"{prefixo}_quali",
    )
    coluna = VARIAVEIS_QUALITATIVAS[variavel_quali]
    base_ordem = referencia if comparando else df
    ordem = ordem_niveis(base_ordem, coluna)
    tabela = tabela_frequencias(df, coluna, ordem)

    col_graf, col_tab = st.columns([3, 2])
    with col_graf:
        if comparando:
            tabela_ref = tabela_frequencias(referencia, coluna, ordem)
            st.altair_chart(grafico_frequencias_comparadas(tabela, tabela_ref), alt=f"Proporções por {variavel_quali.lower()}")
        else:
            st.altair_chart(grafico_frequencias(tabela, conjunto), alt=f"Proporções por {variavel_quali.lower()}")
    with col_tab:
        exibicao = tabela.copy()
        exibicao["Proporção"] = exibicao["Proporção"].map(lambda v: fmt_pct(v, 2))
        if comparando:
            exibicao["Na população"] = tabela_ref["Proporção"].map(lambda v: fmt_pct(v, 2))
            diferenca = (tabela["Proporção"] - tabela_ref["Proporção"]) * 100
            exibicao["Diferença (p.p.)"] = diferenca.map(lambda v: f"{'+' if v > 0 else ''}{fmt_num(v, 2)}")
        exibicao["Frequência"] = exibicao["Frequência"].map(fmt_num)
        total = {"Nível": "Total", "Frequência": fmt_num(len(df)), "Proporção": "100,00%"}
        if comparando:
            total |= {"Na população": "100,00%", "Diferença (p.p.)": ""}
        exibicao = pd.concat([exibicao, pd.DataFrame([total])], ignore_index=True)
        st.dataframe(exibicao, hide_index=True, alt=f"Tabela de frequências de {variavel_quali.lower()}")


def tabela_descritivas(valores, valores_ref, conjunto):
    medidas = descritivas(valores)
    linhas = {"Medida": list(medidas), conjunto: [formatar_medida(k, v) for k, v in medidas.items()]}
    if valores_ref is not None:
        medidas_ref = descritivas(valores_ref)
        linhas[POPULACAO] = [formatar_medida(k, v) for k, v in medidas_ref.items()]
        # Diferença relativa não faz sentido para n e para o mínimo (centavos na população)
        linhas["Diferença relativa"] = [
            "" if k in ("n", "Mínimo") else fmt_pct((medidas[k] - medidas_ref[k]) / abs(medidas_ref[k])) for k in medidas
        ]
    return pd.DataFrame(linhas)


def formatar_medida(nome, valor):
    if nome == "n":
        return fmt_num(valor)
    if nome == "Coef. de variação":
        return fmt_pct(valor)
    if nome in ("Assimetria", "Curtose (excesso)"):
        return fmt_num(valor, 2)
    return fmt_reais(valor)


def filtros_barra_lateral(chave):
    st.sidebar.header("Filtros")
    regioes = st.sidebar.pills(
        "Região do parlamentar", REGIOES, selection_mode="multi", default=REGIOES, key=f"{chave}_regioes",
        help="Filtra os lançamentos pela região da UF do deputado.",
    )
    st.sidebar.caption(FONTE)
    return regioes


# ===========================================================================
# Hipóteses
# ===========================================================================

def _evento_combustivel(df):
    return df["categoria"] == "Combustíveis"


def _evento_sudeste(df):
    return df["regiao"] == "Sudeste"


def _todas(df):
    return pd.Series(True, index=df.index)


def _passagem_aerea(df):
    return df["categoria"] == "Passagem aérea"


HIPOTESES = [
    {
        "id": "P1",
        "tipo": "proporcao",
        "titulo": "Proporção de lançamentos de combustível",
        "pergunta": "Mais de 30% dos lançamentos da cota são gastos com combustível?",
        "variavel": "Categoria de despesa (qualitativa nominal)",
        "evento": "o lançamento é da categoria Combustíveis e lubrificantes",
        "parametro": "p = proporção de lançamentos de combustível",
        "simbolo": "p",
        "simbolo_texto": "p",
        "valor_h0": 0.30,
        "alternativa": "maior",
        "justificativa": (
            "Combustível é a despesa mais frequente da cota: são notas pequenas e numerosas, "
            "ligadas ao deslocamento terrestre do deputado na base eleitoral. Saber se elas passam de "
            "30% dos lançamentos (quase 1 em cada 3) indica onde se concentra o trabalho de conferência "
            "das notas. O teste é unilateral à direita porque a pergunta é se a proporção supera 30%."
        ),
        "funcao": _evento_combustivel,
    },
    {
        "id": "P2",
        "tipo": "proporcao",
        "titulo": "Proporção de lançamentos do Sudeste",
        "pergunta": "Os deputados do Sudeste concentram mais lançamentos que o peso da sua bancada na Câmara?",
        "variavel": "Região do parlamentar (qualitativa nominal)",
        "evento": "o lançamento é de um deputado eleito por ES, MG, RJ ou SP",
        "parametro": "p = proporção de lançamentos de deputados do Sudeste",
        "simbolo": "p",
        "simbolo_texto": "p",
        # 179 das 513 cadeiras da Câmara são do Sudeste (SP 70, MG 53, RJ 46, ES 10)
        "valor_h0": 179 / 513,
        "alternativa": "maior",
        "justificativa": (
            "O Sudeste elege 179 das 513 cadeiras da Câmara (34,9%). Se cada deputado usasse a cota com a "
            "mesma frequência, os lançamentos do Sudeste seriam 34,9% do total. O teste unilateral à direita "
            "verifica se a região lança mais despesas do que o tamanho da sua bancada explicaria, por exemplo "
            "por ter mais deputados com escritórios e deslocamentos frequentes na base eleitoral."
        ),
        "funcao": _evento_sudeste,
    },
    {
        "id": "M1",
        "tipo": "media",
        "titulo": "Valor médio por lançamento",
        "pergunta": "O valor médio de um lançamento da cota passa de R$ 1.000?",
        "variavel": "Valor líquido da despesa, em R$ (quantitativa contínua)",
        "evento": "todos os lançamentos",
        "parametro": "μ = valor líquido médio por lançamento",
        "simbolo": "\\mu",
        "simbolo_texto": "μ",
        "valor_h0": 1000.0,
        "alternativa": "maior",
        "justificativa": (
            "A maior parte dos lançamentos é de valor baixo (combustível, táxi, alimentação), mas existem "
            "despesas de dezenas de milhares de reais (divulgação, fretamento de aeronaves). R$ 1.000 é uma "
            "referência redonda para avaliar se essas despesas altas puxam a média para cima. "
            "O teste é unilateral à direita."
        ),
        "funcao": _todas,
    },
    {
        "id": "M2",
        "tipo": "media",
        "titulo": "Valor médio da passagem aérea",
        "pergunta": "O valor médio de uma passagem aérea paga pela cota passa de R$ 1.000?",
        "variavel": "Valor líquido das passagens aéreas, em R$ (quantitativa contínua)",
        "evento": "lançamentos da categoria Passagem aérea (SIGEPA, RPA e reembolso)",
        "parametro": "μ = valor líquido médio de uma passagem aérea",
        "simbolo": "\\mu",
        "simbolo_texto": "μ",
        "valor_h0": 1000.0,
        "alternativa": "maior",
        # Hipótese nula escrita como igualdade, como na formulação do trabalho
        "sinal_h0": "=",
        "justificativa": (
            "Passagem aérea é o segundo tipo de lançamento mais frequente da cota. A pergunta investiga se "
            "uma passagem é, individualmente, uma despesa elevada, usando a mesma referência de R$ 1.000 "
            "de M1. Assim dá para comparar as passagens com o lançamento médio de todas as despesas. O teste "
            "é unilateral à direita. A análise usa só as passagens aéreas da amostra: dentro desse subgrupo "
            "a amostra continua aleatória, mas o tamanho dela também é aleatório."
        ),
        "funcao": _passagem_aerea,
    },
]
HIPOTESES_POR_ID = {h["id"]: h for h in HIPOTESES}

SINAIS_H1 = {"maior": ">", "menor": "<", "diferente": "\\neq"}
SINAIS_H0 = {"maior": "\\le", "menor": "\\ge", "diferente": "="}
DESCRICAO_TESTE = {"maior": "unilateral à direita", "menor": "unilateral à esquerda", "diferente": "bilateral"}


def latex_hipoteses(hip):
    # Proporções com até 3 casas, sem zeros à direita (0,3 e 0,349)
    valor = f"{hip['valor_h0']:.3f}".rstrip("0").replace(".", ",") if hip["tipo"] == "proporcao" else fmt_num(hip["valor_h0"], 0)
    valor = valor.replace(",", "{,}")
    s = hip["simbolo"]
    sinal_h0 = hip.get("sinal_h0", SINAIS_H0[hip["alternativa"]])
    return (
        f"H_0: {s} {sinal_h0} {valor} \\qquad "
        f"H_1: {s} {SINAIS_H1[hip['alternativa']]} {valor}"
    )


def analisar(hip, df, alfa):
    """Executa IC e teste da hipótese com os dados de `df` (sempre a amostra)."""
    selecao = hip["funcao"](df)
    if hip["tipo"] == "proporcao":
        return inferencia_proporcao(int(selecao.sum()), len(df), hip["valor_h0"], hip["alternativa"], alfa, 1 - alfa)
    return inferencia_media(df.loc[selecao, "vlrLiquido"], hip["valor_h0"], hip["alternativa"], alfa, 1 - alfa)


def parametro_populacional(hip, populacao):
    selecao = hip["funcao"](populacao)
    if hip["tipo"] == "proporcao":
        return selecao.mean()
    return populacao.loc[selecao, "vlrLiquido"].mean()


def h0_verdadeira(hip, parametro):
    if hip["alternativa"] == "maior":
        return parametro <= hip["valor_h0"]
    if hip["alternativa"] == "menor":
        return parametro >= hip["valor_h0"]
    return np.isclose(parametro, hip["valor_h0"])


# ===========================================================================
# Simulação de muitas AAS (variabilidade amostral)
# ===========================================================================

@st.cache_data(show_spinner="Sorteando 1.000 amostras da população...")
def simular_amostras(id_hipotese, alfa, n, repeticoes=1000, semente=7):
    """Repete a AAS de tamanho n e devolve estimativa, IC e decisão de cada repetição."""
    hip = HIPOTESES_POR_ID[id_hipotese]
    populacao = carregar_populacao()
    rng = np.random.default_rng(semente)
    indices = np.stack([rng.choice(len(populacao), size=n, replace=False) for _ in range(repeticoes)])
    selecao = hip["funcao"](populacao).to_numpy()[indices]

    if hip["tipo"] == "proporcao":
        estimativa = selecao.mean(axis=1)
        z_crit = stats.norm.ppf(1 - alfa / 2)
        margem = z_crit * np.sqrt(estimativa * (1 - estimativa) / n)
        p0 = hip["valor_h0"]
        estatistica = (estimativa - p0) / np.sqrt(p0 * (1 - p0) / n)
        p_valor = calcular_p_valor(stats.norm, estatistica, hip["alternativa"])
        tamanho = np.full(repeticoes, n)
    else:
        valores = populacao["vlrLiquido"].to_numpy()[indices]
        tamanho = selecao.sum(axis=1)
        estimativa = (valores * selecao).sum(axis=1) / tamanho
        desvios = ((valores - estimativa[:, None]) ** 2 * selecao).sum(axis=1)
        erro_padrao = np.sqrt(desvios / (tamanho - 1)) / np.sqrt(tamanho)
        dist = stats.t(tamanho - 1)
        margem = dist.ppf(1 - alfa / 2) * erro_padrao
        estatistica = (estimativa - hip["valor_h0"]) / erro_padrao
        p_valor = calcular_p_valor(dist, estatistica, hip["alternativa"])

    parametro = parametro_populacional(hip, populacao)
    resultado = pd.DataFrame({
        "Amostra": np.arange(1, repeticoes + 1),
        "n": tamanho,
        "Estimativa": estimativa,
        "IC inferior": estimativa - margem,
        "IC superior": estimativa + margem,
        "p-valor": p_valor,
    })
    resultado["Rejeita H0"] = resultado["p-valor"] < alfa
    resultado["Contém o parâmetro"] = (resultado["IC inferior"] <= parametro) & (parametro <= resultado["IC superior"])
    return resultado
