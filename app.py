import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Ponte de segredo: Streamlit Cloud usa st.secrets; local usa .env
try:
    if "DATABASE_URL" in st.secrets and "DATABASE_URL" not in os.environ:
        os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

from db import read_sql
from previsao import PESO_TORNEIO_COPA, carregar_modelos, prever_jogo
from monte_carlo import NOMES_RODADA, preparar, simular_torneio_detalhado
from bandeiras import com_bandeira

st.set_page_config(page_title="IAPredict — Copa 2026", layout="wide")
TOP_N = 12


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@st.cache_resource
def _preparar():
    return preparar()


@st.cache_resource
def _modelos():
    return carregar_modelos()


@st.cache_data
def _probabilidades():
    return read_sql("SELECT * FROM gold_probabilidades_copa ORDER BY prob_campea DESC")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

pagina = st.sidebar.radio(
    "Pagina",
    ["Probabilidades", "Simulacao ao vivo", "Explorador de partidas"],
)

# ---------------------------------------------------------------------------
# Página 1 — Probabilidades pré-computadas
# ---------------------------------------------------------------------------

if pagina == "Probabilidades":
    st.title("Probabilidades — Copa 2026")
    st.caption("Baseado em 1 000 simulacoes Monte Carlo pre-computadas.")

    df = _probabilidades().head(TOP_N).copy()
    df["selecao_flag"] = df["selecao"].apply(com_bandeira)
    df["pct"] = (df["prob_campea"] * 100).round(1)

    chart = (
        alt.Chart(df)
        .mark_bar(color="#1f77b4")
        .encode(
            x=alt.X("pct:Q", title="Probabilidade de titulo (%)"),
            y=alt.Y("selecao_flag:N", sort="-x", title=""),
            tooltip=[
                alt.Tooltip("selecao_flag:N", title="Selecao"),
                alt.Tooltip("pct:Q", title="Prob. titulo (%)"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Detalhamento por fase")
    tabela = df[["selecao_flag", "prob_grupo", "prob_oitavas", "prob_quartas",
                 "prob_semi", "prob_final", "prob_campea"]].copy()
    for col in tabela.columns[1:]:
        tabela[col] = (tabela[col] * 100).round(1).astype(str) + "%"
    tabela = tabela.rename(columns={
        "selecao_flag": "Selecao",
        "prob_grupo":   "Grupos",
        "prob_oitavas": "Oitavas",
        "prob_quartas": "Quartas",
        "prob_semi":    "Semis",
        "prob_final":   "Final",
        "prob_campea":  "Campea",
    })
    st.dataframe(tabela, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Página 2 — Simulação ao vivo
# ---------------------------------------------------------------------------

elif pagina == "Simulacao ao vivo":
    st.title("Simulacao ao vivo — Copa 2026")
    st.caption("Cada clique roda uma simulacao completa do torneio.")

    if st.button("Simular torneio"):
        dados = _preparar()
        with st.spinner("Simulando..."):
            res = simular_torneio_detalhado(dados)

        # Podio
        col1, col2, col3 = st.columns(3)
        col1.metric("Campea", com_bandeira(res["campeao"]) if res["campeao"] else "-")
        col2.metric("Vice", com_bandeira(res["vice"]) if res["vice"] else "-")
        col3.metric("3 lugar", com_bandeira(res["terceiro"]) if res["terceiro"] else "-")

        st.divider()

        # Mata-mata por rodada
        st.subheader("Mata-mata")
        for rnd, nome in NOMES_RODADA.items():
            jogos = res["mata_mata"].get(rnd, [])
            if not jogos:
                continue
            st.markdown(f"**{nome}**")
            linhas = []
            for j in jogos:
                pen = " (pen.)" if j["penaltis"] else ""
                home_flag = com_bandeira(j["home"])
                away_flag = com_bandeira(j["away"])
                placar = f"{j['gols_home']} – {j['gols_away']}{pen}"
                destaque = "**" if j["vencedor"] == j["home"] else ""
                destaque_v = "**" if j["vencedor"] == j["away"] else ""
                linhas.append({
                    "Casa": f"{destaque}{home_flag}{destaque}",
                    "Placar": placar,
                    "Visitante": f"{destaque_v}{away_flag}{destaque_v}",
                })
            st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

        st.divider()

        # Fase de grupos
        st.subheader("Fase de grupos")
        cols = st.columns(3)
        for i, (grupo, df_g) in enumerate(sorted(res["grupos"].items())):
            with cols[i % 3]:
                st.markdown(f"**Grupo {grupo}**")
                df_g = df_g.copy()
                df_g["selecao"] = df_g["selecao"].apply(com_bandeira)
                df_g = df_g.rename(columns={
                    "posicao": "Pos", "selecao": "Selecao", "jogos": "J",
                    "vitorias": "V", "empates": "E", "derrotas": "D",
                    "gols_pro": "GP", "gols_contra": "GC",
                    "saldo_gols": "SG", "pontos": "Pts",
                })
                st.dataframe(df_g, use_container_width=True, hide_index=True)
    else:
        st.info("Clique em 'Simular torneio' para rodar uma simulacao completa.")

# ---------------------------------------------------------------------------
# Página 3 — Explorador de partidas
# ---------------------------------------------------------------------------

else:
    st.title("Explorador de partidas")
    st.caption("Escolha dois times e veja o xG e as probabilidades de resultado.")

    modelos, colunas, elos = _modelos()
    lista = sorted(elos.keys())

    col1, col2 = st.columns(2)
    with col1:
        casa = st.selectbox("Time da casa", lista, format_func=com_bandeira,
                            index=lista.index("Brazil") if "Brazil" in lista else 0)
    with col2:
        visit = st.selectbox("Visitante", lista, format_func=com_bandeira,
                             index=lista.index("Argentina") if "Argentina" in lista else 1)

    neutro = st.checkbox("Campo neutro")

    if st.button("Prever"):
        r = prever_jogo(casa, visit, neutro, PESO_TORNEIO_COPA, modelos, colunas, elos)

        c1, c2 = st.columns(2)
        c1.metric(f"xG {com_bandeira(casa)}", f"{r['gols_esperados_casa']:.2f}")
        c2.metric(f"xG {com_bandeira(visit)}", f"{r['gols_esperados_visitante']:.2f}")

        st.subheader("Probabilidades")
        prob_df = pd.DataFrame({
            "Resultado": ["Vitoria casa", "Empate", "Vitoria visitante"],
            "Probabilidade": [
                r["prob_vitoria"], r["prob_empate"], r["prob_derrota"]
            ],
        })
        chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X("Resultado:N", sort=None),
                y=alt.Y("Probabilidade:Q", axis=alt.Axis(format=".0%")),
                color=alt.Color("Resultado:N", legend=None),
                tooltip=[
                    alt.Tooltip("Resultado:N"),
                    alt.Tooltip("Probabilidade:Q", format=".1%"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
