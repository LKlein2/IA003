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
)
from estatistica import ic_bootstrap_media

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



def fmt_ic(hip, ic):
    return f"[{fmt_variavel(ic[0], eh_proporcao(hip))}; {fmt_variavel(ic[1], eh_proporcao(hip))}]"


def fmt_h0(hip):
    """Valor de referência de H0: proporção com 1 casa (34,9%), média sem centavos."""
    return fmt_variavel(hip["valor_h0"], eh_proporcao(hip), 1 if eh_proporcao(hip) else 0)


def contem(ic, valor):
    return ic[0] <= valor <= ic[1]


def tipo_de_decisao(hip, res, parametro):
    """Compara a decisão do teste com a verdade conhecida na população."""
    # h0_verdadeira é o complemento de H1, mesmo quando H0 é escrita como igualdade
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
        f"({fmt_variavel(parametro, prop)}) e o valor de H₀ ({fmt_h0(hip)}) é de {distancia}, e a amostra "
        f"(n = {fmt_num(res['n'])}) não teve precisão suficiente para detectá-la."
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
    titulo_x = "Proporção" if prop else "Valor médio (R$)"
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
        "n": res["n"],
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
    st.metric("Tamanho da amostra (n)", fmt_num(res["n"]), border=True)
    st.metric("Estimativa pontual", fmt_variavel(res["estimativa"], prop), border=True,
              help="Proporção amostral p̂." if prop else f"Média amostral x̄ (desvio-padrão s = {fmt_reais(res['desvio_padrao'])}).")
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
    relativo = f", ou {fmt_pct(diferenca / parametro)} do parâmetro"
    with st.container(border=True):
        st.markdown("**Comparação com a população**")
        st.markdown(
            f"- Parâmetro da população: **{fmt_variavel(parametro, prop)}**\n"
            f"- Estimativa da amostra: **{fmt_variavel(res['estimativa'], prop)}** "
            f"(diferença de {texto_diferenca}{relativo})\n"
            f"- O IC {'**contém**' if contem(res['ic'], parametro) else '**não contém**'} o parâmetro populacional.\n"
            f"- {tipo_de_decisao(hip, res, parametro)}"
        )

# ===========================================================================
# Conclusão
# ===========================================================================

st.subheader("Conclusão", divider="gray")


def frase_decisao(hip):
    res = resultados[hip["id"]]
    if res["rejeita_h0"]:
        return f"**{hip['id']}**: rejeitamos H0. Há evidência de que {hip['h1_texto']}."
    return f"**{hip['id']}**: não rejeitamos H0. Não há evidência de que {hip['h1_texto']}."


def frase_diferenca(hip):
    prop = eh_proporcao(hip)
    diferenca = resultados[hip["id"]]["estimativa"] - parametros[hip["id"]]
    if prop:
        return f"{'+' if diferenca > 0 else '−'}{fmt_num(abs(diferenca) * 100, 1)} p.p. em {hip['id']}"
    return f"{'+' if diferenca > 0 else '−'}{fmt_reais(abs(diferenca), 0)} em {hip['id']}"


def frase_margem(hip):
    margem = resultados[hip["id"]]["margem"]
    if eh_proporcao(hip):
        return f"±{fmt_num(margem * 100, 1)} p.p. em {hip['id']}"
    return f"±{fmt_reais(margem, 0)} em {hip['id']}"


contem_todos = all(contem(resultados[h["id"]]["ic"], parametros[h["id"]]) for h in HIPOTESES)
erros = [h for h in HIPOTESES if tipo_de_decisao(h, resultados[h["id"]], parametros[h["id"]]).startswith("Erro")]
# Conclusões com p-valor perto de α são as que mais mudariam com outra amostra
sensiveis = [h["id"] for h in HIPOTESES if alfa / 5 <= resultados[h["id"]]["p_valor"] <= 2 * alfa]

texto_ic = (
    "Nos quatro casos o parâmetro da população ficou **dentro** do intervalo de confiança."
    if contem_todos
    else "Em algum caso o parâmetro da população ficou **fora** do intervalo de confiança, o que acontece "
    f"em cerca de {fmt_pct(alfa, 0)} das amostras."
)
if erros:
    texto_erros = " ".join(
        f"Em {h['id']} houve {tipo_de_decisao(h, resultados[h['id']], parametros[h['id']])[0].lower()}"
        f"{tipo_de_decisao(h, resultados[h['id']], parametros[h['id']])[1:]}"
        for h in erros
    )
else:
    texto_erros = "Todas as decisões dos testes coincidem com o que é verdade na população."
texto_sensiveis = (
    f" As conclusões com p-valor mais perto de α ({', '.join(sensiveis)}) são as que mais poderiam mudar com outra amostra."
    if sensiveis
    else ""
)

with st.container(border=True):
    st.markdown(
        f"""
**Resultados dos testes** (α = {fmt_pct(alfa, 0)})

{chr(10).join(f"- {frase_decisao(h)}" for h in HIPOTESES)}

**Comparação com a população.** As estimativas ficaram próximas dos parâmetros: diferença de
{", ".join(frase_diferenca(h) for h in HIPOTESES)}. {texto_ic} {texto_erros}

**Efeito da variabilidade amostral.** A estimativa não coincide com o parâmetro porque cada amostra aleatória
é diferente. O intervalo de confiança mede essa incerteza: a margem de erro foi de
{", ".join(frase_margem(h) for h in HIPOTESES)}.{texto_sensiveis}

**Limitações.** Os valores das despesas são muito assimétricos: poucas despesas muito altas puxam a média
para cima e alargam os intervalos. Só {fmt_num(resultados['M2']['n'])} passagens aéreas entraram na amostra:
com tão poucos dados, o resultado de M2 depende muito de quais passagens foram sorteadas. E a unidade amostral é o lançamento, não o deputado: quem lança
mais despesas pesa mais nos resultados.
"""
    )
