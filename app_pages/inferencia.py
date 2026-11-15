import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from comum import (
    AMOSTRA,
    COR_ACEITACAO,
    COR_H0,
    COR_REJEICAO,
    CORES_CONJUNTO,
    DESCRICAO_TESTE,
    EIXO_DECIMAL,
    EIXO_PCT,
    EIXO_REAIS,
    HIPOTESES,
    HIPOTESES_POR_ID,
    POPULACAO,
    analisar,
    carregar_amostra,
    carregar_populacao,
    fmt_num,
    fmt_p_valor,
    fmt_pct,
    fmt_reais,
    fmt_variavel,
    h0_verdadeira,
    latex_hipoteses,
    parametro_populacional,
    simular_amostras,
)
from estatistica import ic_bootstrap_diferenca, ic_bootstrap_media

amostra = carregar_amostra()
populacao = carregar_populacao()

st.caption(
    f"Os intervalos de confiança e os testes usam **somente a amostra** (n = {fmt_num(len(amostra))}). "
    f"Os parâmetros da população (N = {fmt_num(len(populacao))}) são calculados diretamente e servem para "
    "conferir as conclusões, coisa que numa pesquisa real não seria possível."
)

alfa = st.segmented_control(
    "Nível de significância",
    [0.01, 0.05, 0.10],
    default=0.05,
    required=True,
    format_func=lambda a: f"α = {fmt_pct(a, 0)} (IC de {fmt_pct(1 - a, 0)})",
    help="O nível de confiança dos intervalos acompanha o nível de significância: confiança = 1 − α.",
)
confianca = fmt_pct(1 - alfa, 0)

resultados = {h["id"]: analisar(h, amostra, alfa) for h in HIPOTESES}
parametros = {h["id"]: parametro_populacional(h, populacao) for h in HIPOTESES}


# ===========================================================================
# Funções de exibição
# ===========================================================================

def eh_proporcao(hip):
    return hip["tipo"] == "proporcao"


def eh_diferenca(hip):
    return hip["tipo"] == "diferenca_medias"


def grupos(hip, df):
    """Valores dos dois grupos comparados numa hipótese de diferença de médias."""
    return df.loc[hip["funcao"](df), "vlrLiquido"], df.loc[hip["funcao_grupo2"](df), "vlrLiquido"]


def fmt_ic(hip, ic):
    return f"[{fmt_variavel(ic[0], eh_proporcao(hip))}; {fmt_variavel(ic[1], eh_proporcao(hip))}]"


def fmt_h0(hip):
    """Valor de referência de H0: proporção com 1 casa (34,9%), média sem centavos."""
    return fmt_variavel(hip["valor_h0"], eh_proporcao(hip), 1 if eh_proporcao(hip) else 0)


def contem(ic, valor):
    return ic[0] <= valor <= ic[1]


def tipo_de_decisao(hip, res, parametro):
    """Compara a decisão do teste com a verdade conhecida na população."""
    # h0_verdadeira é o complemento de H1 (para M2, H1 falsa = média das passagens não supera a das demais)
    verdadeira = h0_verdadeira(hip, parametro)
    if res["rejeita_h0"] and not verdadeira:
        return "Decisão correta: rejeitou H₀, e H₁ é verdadeira na população."
    if not res["rejeita_h0"] and verdadeira:
        return "Decisão correta: não rejeitou H₀, e H₁ é falsa na população."
    if res["rejeita_h0"]:
        return "Erro tipo I: rejeitou H₀, mas H₁ é falsa na população."
    prop = eh_proporcao(hip)
    distancia = abs(parametro - hip["valor_h0"])
    distancia = f"{fmt_num(distancia * 100, 2)} p.p." if prop else fmt_reais(distancia)
    return (
        "Erro tipo II: não rejeitou uma H₀ que é falsa na população. A diferença real entre o parâmetro "
        f"({fmt_variavel(parametro, prop)}) e o valor de H₀ ({fmt_h0(hip)}) é de só {distancia}, "
        f"pequena demais para ser detectada com n = {fmt_num(res['n'])}."
    )


def grafico_resumo(hips):
    """IC de cada hipótese (laranja), estimativa da amostra, parâmetro populacional e valor de H0."""
    linhas = []
    for hip in hips:
        res = resultados[hip["id"]]
        prop = eh_proporcao(hip)
        linhas.append({
            "Hipótese": f"{hip['id']} - {hip['titulo']}",
            "IC inferior": res["ic"][0], "IC superior": res["ic"][1],
            "Estimativa": res["estimativa"], "Parâmetro": parametros[hip["id"]], "H0": hip["valor_h0"],
            "IC": fmt_ic(hip, res["ic"]),
            "Estimativa (texto)": fmt_variavel(res["estimativa"], prop),
            "Parâmetro (texto)": fmt_variavel(parametros[hip["id"]], prop),
            "H0 (texto)": fmt_variavel(hip["valor_h0"], prop),
        })
    dados = pd.DataFrame(linhas)
    prop = eh_proporcao(hips[0])
    eixo = alt.Axis(labelExpr=EIXO_PCT if prop else EIXO_REAIS)
    titulo_x = "Proporção" if prop else "R$ (média em M1, diferença de médias em M2)"
    y = alt.Y("Hipótese:N", title=None, axis=alt.Axis(labelLimit=260))
    tooltip = [
        alt.Tooltip("Hipótese:N"),
        alt.Tooltip("Estimativa (texto):N", title="Estimativa (amostra)"),
        alt.Tooltip("IC:N", title=f"IC {confianca}"),
        alt.Tooltip("Parâmetro (texto):N", title="Parâmetro (população)"),
        alt.Tooltip("H0 (texto):N", title="Valor de H0"),
    ]
    base = alt.Chart(dados).encode(y=y, tooltip=tooltip)
    ic = base.mark_rule(strokeWidth=3, color=CORES_CONJUNTO[AMOSTRA]).encode(
        x=alt.X("IC inferior:Q", title=titulo_x, axis=eixo, scale=alt.Scale(zero=False)), x2="IC superior:Q"
    )
    estimativa = base.mark_circle(size=120, opacity=1, color=CORES_CONJUNTO[AMOSTRA], stroke="white", strokeWidth=1.5).encode(x="Estimativa:Q")
    parametro = base.mark_point(shape="diamond", size=140, filled=True, color=CORES_CONJUNTO[POPULACAO], stroke="white", strokeWidth=1.5).encode(x="Parâmetro:Q")
    h0 = base.mark_tick(thickness=2, size=30, color=COR_H0, strokeDash=[4, 3]).encode(x="H0:Q")
    return (h0 + ic + estimativa + parametro).properties(height=150)


def grafico_estatistica(res):
    """Distribuição da estatística de teste sob H0, região de rejeição e valor observado."""
    dist = res["distribuicao"]
    observado = res["estatistica"]
    limite = max(4.0, abs(observado) + 0.8)
    x = np.linspace(-limite, limite, 400)
    curva = pd.DataFrame({"x": x, "densidade": dist.pdf(x)})
    criticos = res["critico"]
    if len(criticos) == 2:
        rejeita = (x <= criticos[0]) | (x >= criticos[1])
    elif criticos[0] > 0:
        rejeita = x >= criticos[0]
    else:
        rejeita = x <= criticos[0]
    curva["Região"] = np.where(rejeita, "Rejeição de H0", "Não rejeição de H0")
    # Área de rejeição pintada só onde a região vale; fora dela a densidade vira zero
    curva["densidade_rejeicao"] = np.where(rejeita, curva["densidade"], 0)

    nome = res["nome_estatistica"]
    eixo_x = alt.X("x:Q", title=f"Estatística {nome}", axis=alt.Axis(labelExpr=EIXO_DECIMAL))
    area = alt.Chart(curva).mark_area(color=COR_ACEITACAO, opacity=0.25).encode(x=eixo_x, y=alt.Y("densidade:Q", title="Densidade", axis=None))
    rejeicao = alt.Chart(curva).mark_area(color=COR_REJEICAO, opacity=0.55).encode(x="x:Q", y="densidade_rejeicao:Q")
    linha = alt.Chart(curva).mark_line(color=COR_H0, strokeWidth=1.5).encode(x="x:Q", y="densidade:Q")

    marcas = pd.DataFrame({
        "x": [observado] + list(criticos),
        "Rótulo": [f"{nome} observado = {fmt_num(observado, 2)}"] + [f"crítico = {fmt_num(c, 2)}" for c in criticos],
        "Tipo": ["Observado"] + ["Crítico"] * len(criticos),
    })
    regra_obs = alt.Chart(marcas[marcas["Tipo"] == "Observado"]).mark_rule(color=CORES_CONJUNTO[AMOSTRA], strokeWidth=2.5).encode(
        x="x:Q", tooltip=["Rótulo:N"]
    )
    texto_obs = alt.Chart(marcas[marcas["Tipo"] == "Observado"]).mark_text(
        align="left" if observado < limite - 1.5 else "right", dx=5 if observado < limite - 1.5 else -5,
        y=12, color="#52514e", fontWeight="bold",
    ).encode(x="x:Q", text="Rótulo:N")
    regra_crit = alt.Chart(marcas[marcas["Tipo"] == "Crítico"]).mark_rule(color=COR_REJEICAO, strokeDash=[4, 3]).encode(
        x="x:Q", tooltip=["Rótulo:N"]
    )
    return (area + rejeicao + linha + regra_crit + regra_obs + texto_obs).properties(height=260)


def grafico_simulacao_estimativas(sim, hip, parametro):
    prop = eh_proporcao(hip)
    contagem, limites = np.histogram(sim["Estimativa"], bins=30)
    dados = pd.DataFrame({"inicio": limites[:-1], "fim": limites[1:], "Amostras": contagem})
    dados["Intervalo"] = [f"{fmt_variavel(a, prop)} a {fmt_variavel(b, prop)}" for a, b in zip(limites[:-1], limites[1:])]
    eixo = alt.Axis(labelExpr=EIXO_PCT if prop else EIXO_REAIS)
    barras = alt.Chart(dados).mark_rect(color=CORES_CONJUNTO[AMOSTRA], opacity=0.85, stroke="white", strokeWidth=1).encode(
        x=alt.X("inicio:Q", title="Estimativa em cada amostra", axis=eixo, scale=alt.Scale(zero=False)),
        x2="fim:Q",
        y=alt.Y("Amostras:Q", title="Número de amostras"),
        y2=alt.datum(0),
        tooltip=["Intervalo:N", "Amostras:Q"],
    )
    referencias = pd.DataFrame({
        "x": [parametro, hip["valor_h0"], resultados[hip["id"]]["estimativa"]],
        "Referência": ["Parâmetro da população", "Valor de H0", "Nossa amostra"],
    })
    referencias["Valor"] = referencias["x"].map(lambda v: fmt_variavel(v, prop))
    cores = alt.Color(
        "Referência:N",
        scale=alt.Scale(domain=list(referencias["Referência"]), range=[CORES_CONJUNTO[POPULACAO], COR_H0, "#52514e"]),
        legend=alt.Legend(orient="top", title=None),
    )
    tracos = alt.StrokeDash(
        "Referência:N",
        scale=alt.Scale(domain=list(referencias["Referência"]), range=[[1, 0], [5, 4], [2, 2]]),
        legend=None,
    )
    regras = alt.Chart(referencias).mark_rule(strokeWidth=2.5).encode(x="x:Q", color=cores, strokeDash=tracos, tooltip=["Referência:N", "Valor:N"])
    return (barras + regras).properties(height=280)


def grafico_simulacao_intervalos(sim, hip, parametro, quantidade=60):
    prop = eh_proporcao(hip)
    dados = sim.head(quantidade).copy()
    dados["Resultado"] = np.where(dados["Contém o parâmetro"], "Contém o parâmetro", "Não contém o parâmetro")
    dados["IC"] = [f"[{fmt_variavel(a, prop)}; {fmt_variavel(b, prop)}]" for a, b in zip(dados["IC inferior"], dados["IC superior"])]
    eixo = alt.Axis(labelExpr=EIXO_PCT if prop else EIXO_REAIS)
    cor = alt.Color(
        "Resultado:N",
        scale=alt.Scale(domain=["Contém o parâmetro", "Não contém o parâmetro"], range=[COR_ACEITACAO, COR_REJEICAO]),
        legend=alt.Legend(orient="top", title=None),
    )
    intervalos = alt.Chart(dados).mark_rule(strokeWidth=2).encode(
        y=alt.Y("Amostra:O", title=f"{quantidade} primeiras amostras", axis=alt.Axis(labels=False, ticks=False)),
        x=alt.X("IC inferior:Q", title=f"IC de {confianca} em cada amostra", axis=eixo, scale=alt.Scale(zero=False)),
        x2="IC superior:Q",
        color=cor,
        tooltip=["Amostra:O", "IC:N", "n:Q"],
    )
    linha = alt.Chart(pd.DataFrame({"x": [parametro]})).mark_rule(color=CORES_CONJUNTO[POPULACAO], strokeWidth=2).encode(x="x:Q")
    return (intervalos + linha).properties(height=280)


# ===========================================================================
# Resumo das quatro análises
# ===========================================================================

st.subheader("Resumo das quatro análises", divider="gray")
linhas = []
for hip in HIPOTESES:
    res = resultados[hip["id"]]
    parametro = parametros[hip["id"]]
    prop = eh_proporcao(hip)
    simbolo = hip["simbolo_texto"]
    sinal_h0 = "=" if hip.get("sinal_h0") == "=" else {"maior": "≤", "menor": "≥", "diferente": "="}[hip["alternativa"]]
    sinal_h1 = {"maior": ">", "menor": "<", "diferente": "≠"}[hip["alternativa"]]
    linhas.append({
        "Hipótese": hip["id"],
        "Parâmetro": hip["titulo"],
        "H0 · H1": f"{simbolo} {sinal_h0} {fmt_h0(hip)} · {simbolo} {sinal_h1} {fmt_h0(hip)}",
        "n": f"{res['n1']} + {res['n2']}" if eh_diferenca(hip) else str(res["n"]),
        "Estimativa": fmt_variavel(res["estimativa"], prop),
        f"IC {confianca}": fmt_ic(hip, res["ic"]),
        "Estatística": f"{res['nome_estatistica']} = {fmt_num(res['estatistica'], 2)}",
        "p-valor": fmt_p_valor(res["p_valor"]),
        "Decisão": "Rejeita H0" if res["rejeita_h0"] else "Não rejeita H0",
        "Parâmetro (população)": fmt_variavel(parametro, prop),
        "IC contém o parâmetro?": "Sim" if contem(res["ic"], parametro) else "Não",
    })
st.dataframe(pd.DataFrame(linhas), hide_index=True, alt="Resumo dos intervalos de confiança e testes de hipóteses")

col_p, col_m = st.columns(2)
with col_p:
    st.markdown("**Proporções**")
    st.altair_chart(grafico_resumo([h for h in HIPOTESES if eh_proporcao(h)]), alt="Intervalos de confiança das proporções")
with col_m:
    st.markdown("**Médias**")
    st.altair_chart(grafico_resumo([h for h in HIPOTESES if not eh_proporcao(h)]), alt="Intervalos de confiança das médias")
st.caption(
    f"Linha laranja = IC de {confianca} calculado com a amostra; círculo laranja = estimativa pontual; "
    "losango azul = parâmetro da população; traço cinza tracejado = valor de referência de H0."
)

# ===========================================================================
# Detalhe de cada hipótese
# ===========================================================================

st.subheader("Detalhe por hipótese", divider="gray")
id_escolhido = st.segmented_control(
    "Hipótese",
    [h["id"] for h in HIPOTESES],
    default="P1",
    required=True,
    format_func=lambda i: f"{i} - {HIPOTESES_POR_ID[i]['titulo']}",
)
hip = HIPOTESES_POR_ID[id_escolhido]
res = resultados[id_escolhido]
parametro = parametros[id_escolhido]
prop = eh_proporcao(hip)

col_def, col_hip = st.columns([3, 2])
with col_def:
    st.markdown(f"#### {hip['pergunta']}")
    st.markdown(
        f"- **Variável:** {hip['variavel']}\n"
        f"- **{'Evento de interesse' if prop else 'Lançamentos analisados'}:** {hip['evento']}\n"
        f"- **Parâmetro:** {hip['parametro']}\n"
        f"- **Justificativa:** {hip['justificativa']}"
    )
with col_hip:
    with st.container(border=True):
        st.markdown(f"**Hipóteses** (teste {DESCRICAO_TESTE[hip['alternativa']]})")
        st.latex(latex_hipoteses(hip))
        if prop:
            st.markdown(
                f"**Procedimento:** teste Z para uma proporção e IC de Wald. "
                f"Estatística: $Z = \\dfrac{{\\hat p - p_0}}{{\\sqrt{{p_0(1-p_0)/n}}}}$. "
                f"Nível de significância α = {fmt_pct(alfa, 0)}."
            )
        elif eh_diferenca(hip):
            st.markdown(
                f"**Procedimento:** teste t de Welch para a diferença entre duas médias independentes e IC t "
                f"de Welch. Estatística: $t = \\dfrac{{\\bar x_P - \\bar x_D}}{{\\sqrt{{s_P^2/n_P + s_D^2/n_D}}}}$, "
                f"com {fmt_num(res['gl'], 1)} graus de liberdade (Welch-Satterthwaite). "
                f"Nível de significância α = {fmt_pct(alfa, 0)}."
            )
        else:
            st.markdown(
                f"**Procedimento:** teste t de Student para uma média (σ desconhecido) e IC t. "
                f"Estatística: $t = \\dfrac{{\\bar x - \\mu_0}}{{s/\\sqrt{{n}}}}$, com {res['gl']} graus de liberdade. "
                f"Nível de significância α = {fmt_pct(alfa, 0)}."
            )

st.markdown("**Verificação das condições**")
condicoes = pd.DataFrame(
    [{"Condição": nome, "Valor": fmt_num(valor, 1), "Atendida": "Sim" if ok else "Não"} for nome, (valor, ok) in res["condicoes"].items()]
)
condicoes.loc[len(condicoes)] = {
    "Condição": "Amostra aleatória simples, n/N < 5% (independência)",
    "Valor": fmt_pct(len(amostra) / len(populacao), 2),
    "Atendida": "Sim",
}
col_cond, col_nota = st.columns([3, 2])
with col_cond:
    st.dataframe(condicoes, hide_index=True, alt="Condições de aplicação do teste")
with col_nota:
    if prop:
        st.caption(
            "Com pelo menos 10 sucessos e 10 fracassos esperados, a distribuição de p̂ é bem aproximada "
            "pela normal (Teorema Central do Limite)."
        )
    elif eh_diferenca(hip):
        bootstrap = ic_bootstrap_diferenca(*grupos(hip, amostra), 1 - alfa)
        st.caption(
            "Os grupos são independentes: vêm da mesma AAS e cada lançamento pertence a um só grupo. "
            f"As variâncias são muito diferentes (s_P = {fmt_reais(res['desvio1'])}, s_D = {fmt_reais(res['desvio2'])}), "
            "por isso o teste de Welch, que não supõe variâncias iguais. As demais despesas são muito "
            "assimétricas, e a regra n > 15·assimetria² verifica se cada grupo é grande o bastante. Como "
            f"verificação, o IC bootstrap percentil (5.000 reamostragens) dá {fmt_ic(hip, bootstrap)}, "
            f"contra {fmt_ic(hip, res['ic'])} do IC de Welch."
        )
    else:
        bootstrap = ic_bootstrap_media(amostra.loc[hip["funcao"](amostra), "vlrLiquido"], 1 - alfa)
        st.caption(
            f"A variável é assimétrica (coeficiente de assimetria = {fmt_num(res['assimetria'], 2)}). "
            "A regra prática n > 15·assimetria² indica se a amostra é grande o bastante para que a média "
            f"amostral seja aproximadamente normal. Como verificação, o IC bootstrap percentil (5.000 "
            f"reamostragens) dá {fmt_ic(hip, bootstrap)}, próximo do IC t {fmt_ic(hip, res['ic'])}."
        )

st.markdown("**Resultados com a amostra**")
with st.container(horizontal=True):
    if eh_diferenca(hip):
        st.metric("Tamanho dos grupos (nP + nD)", f"{res['n1']} + {res['n2']}", border=True)
        ajuda_estimativa = (
            f"x̄P − x̄D = {fmt_reais(res['media1'])} (passagens) − {fmt_reais(res['media2'])} (demais despesas)."
        )
    else:
        st.metric("Tamanho da amostra (n)", fmt_num(res["n"]), border=True)
        ajuda_estimativa = "Proporção amostral p̂." if prop else f"Média amostral x̄ (desvio-padrão s = {fmt_reais(res['desvio_padrao'])})."
    st.metric("Estimativa pontual", fmt_variavel(res["estimativa"], prop), border=True, help=ajuda_estimativa)
    st.metric(f"IC de {confianca}", fmt_ic(hip, res["ic"]), border=True,
              help=f"Estimativa ± margem de erro de {fmt_variavel(res['margem'], prop, 2)}.")
    st.metric(f"Estatística {res['nome_estatistica']}", fmt_num(res["estatistica"], 3), border=True)
    st.metric("p-valor", fmt_p_valor(res["p_valor"]), border=True)

col_graf, col_conc = st.columns([3, 2])
with col_graf:
    st.markdown(f"**Distribuição da estatística {res['nome_estatistica']} se H0 fosse verdadeira**")
    st.altair_chart(grafico_estatistica(res), alt="Distribuição da estatística de teste sob H0")
    st.caption(
        f"Área vermelha = região de rejeição (α = {fmt_pct(alfa, 0)}). Linha laranja = valor observado na amostra."
    )
with col_conc:
    decisao = "rejeitamos H0" if res["rejeita_h0"] else "não rejeitamos H0"
    sentido = {"maior": "maior que", "menor": "menor que", "diferente": "diferente de"}[hip["alternativa"]]
    if eh_diferenca(hip):
        afirmacao = f"a média das passagens aéreas é {sentido} a média das demais despesas"
    else:
        afirmacao = f"o parâmetro é {sentido} {fmt_h0(hip)}"
    evidencia = (
        f"Há evidência estatística, ao nível de {fmt_pct(alfa, 0)}, de que {afirmacao}."
        if res["rejeita_h0"]
        else f"Não há evidência suficiente, ao nível de {fmt_pct(alfa, 0)}, de que {afirmacao}."
    )
    with st.container(border=True):
        st.markdown("**Conclusão do teste**")
        st.markdown(
            f"Como p-valor = {fmt_p_valor(res['p_valor'])} {'<' if res['rejeita_h0'] else '≥'} α = {fmt_pct(alfa, 0)}, "
            f"**{decisao}**. {evidencia}"
        )
        st.markdown(
            f"Com {confianca} de confiança, o parâmetro populacional está entre "
            f"{fmt_variavel(res['ic'][0], prop)} e {fmt_variavel(res['ic'][1], prop)}."
        )

    diferenca = res["estimativa"] - parametro
    if prop:
        texto_diferenca = f"{'+' if diferenca > 0 else ''}{fmt_num(diferenca * 100, 2)} p.p."
    else:
        texto_diferenca = f"{'+' if diferenca > 0 else '−'}{fmt_reais(abs(diferenca))}"
    # Para diferença de médias o parâmetro fica perto de zero, e o erro relativo não faz sentido
    relativo = "" if eh_diferenca(hip) else f", ou {fmt_pct(diferenca / parametro)} do parâmetro"
    with st.container(border=True):
        st.markdown("**Comparação com a população**")
        st.markdown(
            f"- Parâmetro da população: **{fmt_variavel(parametro, prop)}**\n"
            f"- Estimativa da amostra: **{fmt_variavel(res['estimativa'], prop)}** "
            f"(diferença de {texto_diferenca}{relativo})\n"
            f"- O IC {'**contém**' if contem(res['ic'], parametro) else '**não contém**'} o parâmetro populacional.\n"
            f"- {tipo_de_decisao(hip, res, parametro)}"
        )
        if eh_diferenca(hip):
            pop_p, pop_d = grupos(hip, populacao)
            st.markdown(
                f"Na população, a passagem aérea tem média de {fmt_reais(pop_p.mean())} e mediana de "
                f"{fmt_reais(pop_p.median())}; as demais despesas têm média de {fmt_reais(pop_d.mean())}, mas "
                f"mediana de só {fmt_reais(pop_d.median())}. A passagem típica custa bem mais que a despesa "
                "típica, porém a cauda de valores altos das demais categorias (divulgação, fretamentos, "
                "escritório) puxa a média delas para cima e anula a diferença nas médias."
            )

# ===========================================================================
# Variabilidade amostral
# ===========================================================================

st.subheader("Efeito da variabilidade amostral", divider="gray")
st.markdown(
    f"Outra amostra aleatória daria outra estimativa, outro intervalo e talvez outra conclusão. Para medir "
    f"isso, sorteamos **1.000 novas AAS** de tamanho {fmt_num(len(amostra))} da população e repetimos a "
    f"análise de **{hip['id']}** em cada uma."
)
sim = simular_amostras(hip["id"], alfa, len(amostra))
cobertura = sim["Contém o parâmetro"].mean()
taxa_rejeicao = sim["Rejeita H0"].mean()
verdadeira = h0_verdadeira(hip, parametro)

with st.container(horizontal=True):
    st.metric(f"ICs que contêm o parâmetro", fmt_pct(cobertura), border=True,
              help=f"Pela teoria, cerca de {confianca} dos intervalos devem conter o parâmetro.")
    st.metric(
        "Amostras que rejeitam H0", fmt_pct(taxa_rejeicao), border=True,
        help=("Como H1 é verdadeira na população, esta taxa estima o poder do teste."
              if not verdadeira else "Como H1 é falsa na população, esta taxa estima a frequência de erro tipo I."),
    )
    st.metric("Desvio-padrão das estimativas", fmt_variavel(sim["Estimativa"].std(), prop, 2), border=True,
              help="Erro-padrão observado empiricamente nas 1.000 amostras.")
    st.metric("Erro-padrão da nossa amostra", fmt_variavel(res["erro_padrao"], prop, 2), border=True,
              help="Erro-padrão estimado só com a amostra sorteada.")

col_dist, col_ics = st.columns(2)
with col_dist:
    st.markdown("**Distribuição amostral da estimativa**")
    st.altair_chart(grafico_simulacao_estimativas(sim, hip, parametro), alt="Histograma das estimativas nas 1.000 amostras")
with col_ics:
    st.markdown("**Intervalos de confiança de amostras diferentes**")
    st.altair_chart(grafico_simulacao_intervalos(sim, hip, parametro), alt="Intervalos de confiança das primeiras amostras simuladas")

if verdadeira:
    # Quando o parâmetro está do lado oposto ao de H1, rejeitar fica mais raro que α
    comparacao_alfa = (
        f"abaixo de α = {fmt_pct(alfa, 0)}, porque o parâmetro real fica do lado oposto ao que H1 afirma"
        if taxa_rejeicao < alfa - 0.015
        else f"perto de α = {fmt_pct(alfa, 0)}"
    )
    leitura_poder = (
        f"Como H1 é falsa na população, rejeitar H0 é um erro tipo I. Isso ocorreu em "
        f"{fmt_pct(taxa_rejeicao)} das amostras, {comparacao_alfa}."
    )
else:
    leitura_poder = (
        f"Como H1 é verdadeira na população, o ideal é rejeitar H0. Isso ocorreu em {fmt_pct(taxa_rejeicao)} "
        f"das amostras (poder estimado do teste); nas demais {fmt_pct(1 - taxa_rejeicao)} a conclusão "
        "seria um erro tipo II."
    )
if cobertura >= 1 - alfa - 0.015:
    leitura_cobertura = f"{fmt_pct(cobertura)} dos ICs de {confianca} contêm o parâmetro, perto do nível nominal."
elif prop:
    leitura_cobertura = (
        f"Só {fmt_pct(cobertura)} dos ICs de {confianca} contêm o parâmetro, abaixo do nível nominal. "
        "Com proporções pequenas, o IC de Wald fica curto demais em parte das amostras; um IC de Wilson "
        "corrigiria isso."
    )
else:
    leitura_cobertura = (
        f"Só {fmt_pct(cobertura)} dos ICs de {confianca} contêm o parâmetro, abaixo do nível nominal. "
        "A cauda longa dos valores faz a média amostral demorar a ficar normal: amostras sem valores extremos "
        "subestimam a média e o desvio-padrão ao mesmo tempo, e o IC t fica curto demais."
    )
st.markdown(
    f"**Leitura:** as estimativas se concentram em torno do parâmetro populacional, o que mostra que o "
    f"estimador é não viesado. {leitura_cobertura} {leitura_poder}"
)

# ===========================================================================
# Limitações
# ===========================================================================

st.subheader("Limitações da análise", divider="gray")
st.markdown(
    f"""
- **Uma única amostra.** As conclusões valem para a amostra sorteada com a semente escolhida. A simulação
  acima mostra que outra amostra pode levar a outra conclusão, sobretudo quando o parâmetro está perto do
  valor de H0.
- **Variáveis de valor muito assimétricas.** O valor líquido tem cauda longa (poucos lançamentos de dezenas
  de milhares de reais). Com n = {fmt_num(len(amostra))} a média amostral é aproximadamente normal, mas um
  único valor extremo sorteado desloca bastante a estimativa e alarga o IC. A mediana seria um resumo mais
  estável, mas o trabalho pede inferência sobre a média.
- **Grupos desbalanceados (M2).** Só {fmt_num(resultados['M2']['n1'])} passagens aéreas entraram na amostra,
  contra {fmt_num(resultados['M2']['n2'])} demais despesas, porque a amostra foi dimensionada para a população
  toda e não por categoria. O tamanho de cada grupo também varia de uma amostra para outra. O teste de Welch
  lida com tamanhos e variâncias diferentes, mas o erro-padrão é dominado pelo grupo mais disperso (as
  demais despesas). Além disso, a média esconde que a passagem típica (mediana) custa muito mais que a
  despesa típica: comparar medianas responderia melhor à pergunta "uma passagem custa mais?".
- **Dimensionamento feito para proporção.** O n = {fmt_num(len(amostra))} garante margem de ±5 p.p. para
  proporções. Para as médias, a margem obtida depende da variabilidade dos valores e ficou maior que um
  dimensionamento específico para média daria.
- **Unidade amostral é o lançamento, não o deputado.** Deputados com muitos lançamentos pesam mais nas
  estimativas. Perguntas sobre o comportamento médio por deputado exigiriam outra unidade amostral.
- **Recorte da base.** Lideranças e estornos ficaram fora da população; os resultados não se aplicam a eles.
"""
)
