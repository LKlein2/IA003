# Cota Parlamentar 2025: amostragem e inferência

Trabalho da disciplina IA003. Analisa as despesas da cota parlamentar dos deputados federais em 2025,
compara a população com uma amostra aleatória simples e aplica intervalos de confiança e testes de
hipóteses para duas proporções e duas médias.

## Base de dados

- **Fonte:** Câmara dos Deputados, Dados Abertos. Cota para o Exercício da Atividade Parlamentar (CEAP).
- **Página do conjunto:** https://dadosabertos.camara.leg.br/swagger/api.html?tab=staticfile
- **Arquivo:** https://www.camara.leg.br/cotas/Ano-2025.csv.zip (209.078 registros, 32 variáveis)
- **População:** 200.056 lançamentos de deputados com partido e valor líquido positivo.
- **Unidade amostral:** o lançamento (uma linha do arquivo).

## Amostra

- Tamanho calculado com a fórmula da calculadora da USP/Bauru: confiança de 95%, margem de erro de 5%,
  p = 0,5 e N = 200.056. Resultado: **n = 384**.
- AAS sem reposição com `numpy.random.default_rng(2026)`. A amostra fica em `dados/amostra.csv`.

## Hipóteses (α = 5%)

| | Pergunta | H0 · H1 | Teste |
|---|---|---|---|
| P1 | Mais de 30% dos lançamentos são de combustível? | p ≤ 0,30 · p > 0,30 | Z para proporção |
| P2 | Os deputados do Sudeste concentram mais lançamentos que o peso da bancada (179/513 = 34,9%)? | p ≤ 0,349 · p > 0,349 | Z para proporção |
| M1 | O valor médio de um lançamento passa de R$ 1.000? | μ ≤ 1.000 · μ > 1.000 | t para média |
| M2 | O valor médio de uma passagem aérea passa de R$ 1.000? | μ = 1.000 · μ > 1.000 | t para média |

## Estrutura

```
trabalho_inferencia/
├── streamlit_app.py      <- ponto de entrada (navegação entre as páginas)
├── app_pages/
│   ├── base.py           <- base de dados, dimensionamento e sorteio da AAS
│   ├── populacao.py      <- Painel 1: análise descritiva da população
│   ├── amostra.py        <- Painel 2: análise descritiva da amostra
│   └── inferencia.py     <- Painel 3: inferência e comparação
├── comum.py              <- carga dos dados, gráficos e definição das hipóteses
├── estatistica.py        <- tamanho amostral, ICs e testes
├── preparar_dados.py     <- gera a população e sorteia a amostra
└── dados/
    ├── populacao.parquet
    ├── amostra.csv
    └── metadados.json
```

## Execução local

No PowerShell, a partir desta pasta:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Para regerar a população e a amostra a partir do CSV original da Câmara:

```powershell
python preparar_dados.py C:\caminho\para\Ano-2025.csv
```

## Links dos painéis (após publicar)

Com o app publicado em `https://<seu-app>.streamlit.app`:

- Painel 1 (população): `https://<seu-app>.streamlit.app/populacao`
- Painel 2 (amostra): `https://<seu-app>.streamlit.app/amostra`
- Painel 3 (inferência): `https://<seu-app>.streamlit.app/inferencia`
- Base e amostragem: `https://<seu-app>.streamlit.app/`
