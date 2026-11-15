import streamlit as st

from comum import POPULACAO, carregar_populacao, filtrar, filtros_barra_lateral, fmt_num, painel_descritivo

populacao = carregar_populacao()

st.caption(
    f"Os {fmt_num(len(populacao))} lançamentos de despesa da cota parlamentar em 2025 (deputados com "
    "partido, valor líquido positivo). Os valores deste painel são **parâmetros**: descrevem a população "
    "inteira, sem erro amostral."
)

regioes = filtros_barra_lateral("populacao")
selecao = filtrar(populacao, regioes)

if selecao.empty:
    st.warning("Selecione ao menos uma região na barra lateral.")
    st.stop()

painel_descritivo(selecao, POPULACAO)
