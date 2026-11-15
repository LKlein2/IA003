import streamlit as st

from comum import (
    AMOSTRA,
    CAMINHO_AMOSTRA,
    carregar_amostra,
    carregar_metadados,
    carregar_populacao,
    filtrar,
    filtros_barra_lateral,
    fmt_num,
    painel_descritivo,
)

amostra = carregar_amostra()
populacao = carregar_populacao()
meta = carregar_metadados()

st.caption(
    f"Amostra aleatória simples de {fmt_num(len(amostra))} lançamentos (semente {meta['semente']}). "
    "Os valores deste painel são **estatísticas**: estimativas que mudariam se outra amostra fosse sorteada. "
    "Para comparar, cada gráfico e tabela mostra também o valor da população."
)

regioes = filtros_barra_lateral("amostra")
selecao = filtrar(amostra, regioes)

if selecao.empty:
    st.warning("Selecione ao menos uma região na barra lateral.")
    st.stop()

painel_descritivo(selecao, AMOSTRA, referencia=filtrar(populacao, regioes))

st.subheader("Dados da amostra", divider="gray")
st.dataframe(
    selecao[["id_populacao", "txNomeParlamentar", "sgPartido", "sgUF", "categoria", "datEmissao", "vlrLiquido"]],
    hide_index=True,
    column_config={
        "id_populacao": st.column_config.NumberColumn("Posição na população", format="%d"),
        "txNomeParlamentar": "Parlamentar",
        "sgPartido": "Partido",
        "sgUF": "UF",
        "categoria": "Categoria",
        "datEmissao": st.column_config.DateColumn("Emissão", format="DD/MM/YYYY"),
        "vlrLiquido": st.column_config.NumberColumn("Valor líquido (R$)", format="localized"),
    },
    height=300,
    alt="Lançamentos sorteados na amostra",
)
st.download_button(
    "Baixar amostra (CSV)",
    CAMINHO_AMOSTRA.read_bytes(),
    file_name="amostra.csv",
    mime="text/csv",
    icon=":material/download:",
)
