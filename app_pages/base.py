import pandas as pd
import streamlit as st

from comum import CAMINHO_AMOSTRA, carregar_amostra, carregar_metadados, carregar_populacao, fmt_num, fmt_pct
from estatistica import tamanho_amostral_proporcao

meta = carregar_metadados()
populacao = carregar_populacao()
amostra = carregar_amostra()

st.caption(
    "Trabalho de amostragem, intervalos de confiança e testes de hipóteses sobre os gastos da "
    "cota parlamentar dos deputados federais em 2025."
)

# ---------------------------------------------------------------------------
# Fonte e contexto
# ---------------------------------------------------------------------------
st.subheader("Base de dados", divider="gray")
col_texto, col_numeros = st.columns([3, 2])
with col_texto:
    st.markdown(
        """
**Fonte:** Câmara dos Deputados, Dados Abertos. Cota para o Exercício da Atividade Parlamentar (CEAP).

- Página do conjunto: <https://dadosabertos.camara.leg.br/swagger/api.html?tab=staticfile>
- Arquivo utilizado: <https://www.camara.leg.br/cotas/Ano-2025.csv.zip>

**Contexto:** a CEAP é uma verba mensal que reembolsa despesas ligadas ao mandato de cada deputado
federal, como passagens, combustível, escritório, divulgação e alimentação. O teto mensal varia por UF.
Cada linha do arquivo é um lançamento: um documento fiscal (ou uma parcela dele) apresentado para
reembolso. A Câmara publica os dados por exigência da Lei de Acesso à Informação, para que a sociedade
fiscalize o uso do dinheiro público.
"""
    )
with col_numeros:
    with st.container(horizontal=True):
        st.metric("Registros no arquivo", fmt_num(meta["registros_originais"]), border=True)
        st.metric("Variáveis no arquivo", fmt_num(meta["variaveis_originais"]), border=True)
    with st.container(horizontal=True):
        st.metric("Lançamentos na população", fmt_num(meta["N"]), border=True)
        st.metric("Deputados", fmt_num(populacao["nuDeputadoId"].nunique()), border=True)
    with st.container(horizontal=True):
        st.metric("Partidos", fmt_num(populacao["sgPartido"].nunique()), border=True)
        st.metric("UFs", fmt_num(populacao["sgUF"].nunique()), border=True)

# ---------------------------------------------------------------------------
# População e unidade amostral
# ---------------------------------------------------------------------------
st.subheader("População e unidade amostral", divider="gray")
st.markdown(
    f"""
- **População de interesse:** todos os lançamentos de despesa reembolsados pela CEAP a deputados
  federais filiados a partido em 2025, com valor líquido positivo. São **N = {fmt_num(meta['N'])}** lançamentos.
- **Unidade amostral:** o lançamento (uma linha do arquivo).
- **Recorte aplicado ao arquivo original** (o mesmo usado na disciplina IA001):
    - {fmt_num(meta['removidos_sem_partido'])} lançamentos de lideranças partidárias ou de deputados sem partido;
    - {fmt_num(meta['removidos_valor_nao_positivo'])} lançamentos com valor líquido zero ou negativo
      (estornos e compensações de bilhetes, que não são despesas novas).
"""
)

st.markdown("**Variáveis selecionadas**")
variaveis = pd.DataFrame(
    [
        ("vlrLiquido", "Valor líquido (R$)", "Quantitativa contínua",
         "Valor efetivamente reembolsado ao deputado. Usada nos testes M1 e M2."),
        ("txtDescricao → categoria", "Categoria de despesa", "Qualitativa nominal",
         "Tipo de gasto, agrupado em 10 categorias (as três modalidades de passagem aérea viram uma). Usada em P1 e M2."),
        ("sgUF → regiao", "Região do parlamentar", "Qualitativa nominal", "Região da UF pela qual o deputado foi eleito. Usada em P2."),
    ],
    columns=["Coluna original", "Variável", "Tipo", "Descrição"],
)
st.dataframe(variaveis, hide_index=True, alt="Variáveis selecionadas para a análise")

# ---------------------------------------------------------------------------
# Tamanho da amostra
# ---------------------------------------------------------------------------
st.subheader("Dimensionamento da amostra", divider="gray")
col_formula, col_calc = st.columns([3, 2])
with col_formula:
    st.markdown(
        "Cálculo do tamanho de amostra para estimar uma proporção, com correção para população "
        "finita. É a mesma fórmula da calculadora da USP/Bauru:"
    )
    st.latex(r"n_0 = \frac{z^2 \, p\,(1-p)}{e^2} \qquad\qquad n = \frac{n_0}{1 + \dfrac{n_0 - 1}{N}}")
    st.markdown(
        f"""
| Parâmetro | Valor | Justificativa |
|---|---|---|
| Nível de confiança | {fmt_pct(meta['confianca'], 0)} (z = {fmt_num(meta['z'], 2)}) | Padrão nas ciências sociais e na disciplina. |
| Margem de erro (e) | {fmt_pct(meta['erro'], 0)} | Precisão de ±5 pontos percentuais, suficiente para os testes de proporção planejados (P1 com 30%, P2 com 34,9%). |
| Proporção esperada (p) | {fmt_num(meta['p_planejamento'], 1)} | Sem estimativa prévia confiável, p = 0,5 maximiza p(1−p) e dá o maior n (escolha conservadora). |
| População (N) | {fmt_num(meta['N'])} | Lançamentos após o recorte. |
"""
    )
    st.latex(
        rf"n_0 = \frac{{1{{,}}96^2 \times 0{{,}}5 \times 0{{,}}5}}{{0{{,}}05^2}} = {fmt_num(meta['n0'], 2).replace(',', '{,}')}"
        rf"\qquad n = {fmt_num(meta['n_exato'], 2).replace(',', '{,}')} \;\Rightarrow\; \mathbf{{n = {meta['n']}}}"
    )
    st.markdown(
        f"A fração amostral é n/N = {fmt_pct(meta['n'] / meta['N'], 2)}. Como ela é bem menor que 5%, "
        "a correção para população finita praticamente não altera o resultado, e os intervalos e testes "
        "são calculados com as fórmulas usuais."
    )
with col_calc:
    with st.container(border=True):
        st.markdown("**Simule outros parâmetros**")
        st.caption("Só para explorar: a amostra usada no trabalho continua sendo a de 384 lançamentos.")
        confianca = st.select_slider(
            "Nível de confiança", options=[0.90, 0.95, 0.99], value=0.95, format_func=lambda v: fmt_pct(v, 0)
        )
        erro = st.select_slider(
            "Margem de erro", options=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10], value=0.05,
            format_func=lambda v: fmt_pct(v, 0),
        )
        p = st.slider("Proporção esperada", 0.05, 0.95, 0.5, 0.05)
        simulado = tamanho_amostral_proporcao(meta["N"], confianca, erro, p)
        st.metric("Tamanho da amostra", fmt_num(simulado["n"]))

# ---------------------------------------------------------------------------
# Seleção da AAS
# ---------------------------------------------------------------------------
st.subheader("Seleção da amostra aleatória simples", divider="gray")
st.markdown(
    f"""
1. Cada lançamento da população recebeu um número de posição, de 0 a {fmt_num(meta['N'] - 1)}.
2. Um gerador de números aleatórios (NumPy, `default_rng`) com **semente {meta['semente']}** sorteou
   {meta['n']} posições **sem reposição**. Assim, cada lançamento teve a mesma probabilidade,
   n/N = {fmt_pct(meta['n'] / meta['N'], 3)}, de entrar na amostra, e todo subconjunto de {meta['n']}
   lançamentos tinha a mesma chance de ser o escolhido.
3. Os lançamentos sorteados foram gravados em `amostra.csv`. Com a mesma semente, qualquer pessoa
   reproduz exatamente a mesma amostra rodando `preparar_dados.py`.
"""
)
st.code(
    f"""rng = np.random.default_rng({meta['semente']})
indices = rng.choice(len(populacao), size={meta['n']}, replace=False)
amostra = populacao.iloc[np.sort(indices)]""",
    language="python",
)
with st.container(horizontal=True):
    st.metric("Lançamentos sorteados", fmt_num(len(amostra)), border=True)
    st.metric("Lançamentos repetidos", fmt_num(amostra["id_populacao"].duplicated().sum()), border=True)
    st.metric("Deputados na amostra", fmt_num(amostra["nuDeputadoId"].nunique()), border=True)
    st.metric("Categorias na amostra", fmt_num(amostra["categoria"].nunique()), border=True)
st.download_button(
    "Baixar amostra (CSV)",
    CAMINHO_AMOSTRA.read_bytes(),
    file_name="amostra.csv",
    mime="text/csv",
    icon=":material/download:",
    key="baixar_amostra_base",
)
