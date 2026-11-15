import streamlit as st

st.set_page_config(page_title="Cota Parlamentar 2025 - Inferência", page_icon=":material/query_stats:", layout="wide")

pagina = st.navigation(
    [
        st.Page(
            "app_pages/base.py",
            title="Base e amostragem",
            icon=":material/database:",
            url_path="base",
            default=True,
            help="Fonte, população, variáveis, cálculo do tamanho da amostra e sorteio da AAS.",
        ),
        st.Page(
            "app_pages/populacao.py",
            title="Painel 1 - População",
            icon=":material/groups:",
            url_path="populacao",
            help="Análise descritiva dos 200 mil lançamentos da população.",
        ),
        st.Page(
            "app_pages/amostra.py",
            title="Painel 2 - Amostra",
            icon=":material/filter_alt:",
            url_path="amostra",
            help="Análise descritiva da amostra aleatória simples, comparada com a população.",
        ),
        st.Page(
            "app_pages/inferencia.py",
            title="Painel 3 - Inferência",
            icon=":material/functions:",
            url_path="inferencia",
            help="Intervalos de confiança, testes de hipóteses e comparação com os parâmetros populacionais.",
        ),
    ],
    position="top",
)

st.title(pagina.title, icon=pagina.icon)
pagina.run()
