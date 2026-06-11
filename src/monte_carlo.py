import io
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from db import get_raw_connection, read_sql
from previsao import carregar_artefatos, prever_jogo

load_dotenv()

N_SIMS = 1000
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

NOMES_RODADA = {
    "R32": "32-avos de Final",
    "R16": "Oitavas de Final",
    "QF":  "Quartas de Final",
    "SF":  "Semifinais",
    "3rd": "3º Lugar",
    "Final": "Final",
}

FASES = ["passou_grupos", "oitavas", "quartas", "semi", "final", "campea"]
FASE_PARA_COLUNA = {
    "passou_grupos": "prob_grupo",
    "oitavas":       "prob_oitavas",
    "quartas":       "prob_quartas",
    "semi":          "prob_semi",
    "final":         "prob_final",
    "campea":        "prob_campea",
}

CREATE_SQL = """
CREATE TABLE gold_probabilidades_copa (
    id            bigint generated always as identity primary key,
    selecao       text,
    prob_grupo    double precision,
    prob_oitavas  double precision,
    prob_quartas  double precision,
    prob_semi     double precision,
    prob_final    double precision,
    prob_campea   double precision
)
"""

COPY_SQL = """
COPY gold_probabilidades_copa (selecao, prob_grupo, prob_oitavas, prob_quartas,
                                prob_semi, prob_final, prob_campea)
FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
"""


def carregar_dados():
    copa_df = read_sql(
        "SELECT time_casa, time_visitante, neutro FROM silver_copa2026 ORDER BY id"
    )
    grupos_df = pd.read_csv(os.path.join(DATA_DIR, "grupos_copa2026.csv"))
    calendario_df = pd.read_csv(os.path.join(DATA_DIR, "calendario_copa2026.csv"))

    equipe_para_grupo = dict(zip(grupos_df["nation"], grupos_df["group"]))
    grupos_equipes = grupos_df.groupby("group")["nation"].apply(list).to_dict()

    slots_3 = [
        s for s in pd.concat([calendario_df["home_slot"], calendario_df["away_slot"]])
        if isinstance(s, str) and s.startswith("3")
    ]
    slots_3 = list(dict.fromkeys(slots_3))  # deduplica preservando ordem

    modelos, colunas, elos = carregar_artefatos()
    return copa_df, grupos_equipes, equipe_para_grupo, calendario_df, slots_3, modelos, colunas, elos


def pre_computar_lambdas(copa_df, modelos, colunas, elos):
    cache = {}
    for row in copa_df.itertuples(index=False):
        key = (row.time_casa, row.time_visitante, row.neutro, 3)
        if key not in cache:
            r = prever_jogo(row.time_casa, row.time_visitante, row.neutro, 3,
                            modelos, colunas, elos)
            cache[key] = (r["gols_esperados_casa"], r["gols_esperados_visitante"])
    return cache


def obter_lambda(team_a, team_b, neutro, modelos, colunas, elos, cache):
    key = (team_a, team_b, neutro, 3)
    if key not in cache:
        r = prever_jogo(team_a, team_b, neutro, 3, modelos, colunas, elos)
        cache[key] = (r["gols_esperados_casa"], r["gols_esperados_visitante"])
    return cache[key]


def simular_jogo(lc, lv):
    gc = np.random.poisson(lc)
    gv = np.random.poisson(lv)
    return gc, gv


def classificar_grupo(stats):
    return sorted(
        stats.items(),
        key=lambda x: (-x[1]["pts"], -x[1]["saldo"], -x[1]["gols_pro"], np.random.random()),
    )


def mapear_terceiros(top8, slots_3):
    INF = 1e9
    n = len(top8)
    cost = np.full((n, n), INF)
    for i, (_, grupo) in enumerate(top8):
        for j, slot in enumerate(slots_3):
            if grupo in slot[1:]:
                cost[i, j] = 0.0
    row_ind, col_ind = linear_sum_assignment(cost)
    return {slots_3[col_ind[i]]: top8[row_ind[i]][0] for i in range(n)}


def uma_simulacao(copa_df, grupos_equipes, equipe_para_grupo,
                  calendario_df, slots_3, lambda_cache, modelos, colunas, elos):
    # --- Fase de grupos ---
    standings = {
        g: {e: {"pts": 0, "saldo": 0, "gols_pro": 0} for e in equipes}
        for g, equipes in grupos_equipes.items()
    }

    for row in copa_df.itertuples(index=False):
        lc, lv = lambda_cache[(row.time_casa, row.time_visitante, row.neutro, 3)]
        gc, gv = simular_jogo(lc, lv)

        def upd(equipe, gf, gc_opp):
            s = standings[equipe_para_grupo[equipe]][equipe]
            s["gols_pro"] += gf
            s["saldo"] += gf - gc_opp
            if gf > gc_opp:
                s["pts"] += 3
            elif gf == gc_opp:
                s["pts"] += 1

        upd(row.time_casa, gc, gv)
        upd(row.time_visitante, gv, gc)

    # --- Classificação dos grupos ---
    slot_map = {}
    terceiros = []

    for grupo, stats in standings.items():
        ranking = classificar_grupo(stats)
        slot_map[f"1{grupo}"] = ranking[0][0]
        slot_map[f"2{grupo}"] = ranking[1][0]
        t_equipe, t_stats = ranking[2]
        terceiros.append((t_equipe, grupo,
                          t_stats["pts"], t_stats["saldo"], t_stats["gols_pro"]))

    # --- 8 melhores terceiros ---
    terceiros_sorted = sorted(
        terceiros,
        key=lambda x: (-x[2], -x[3], -x[4], np.random.random()),
    )
    top8 = [(e, g) for e, g, *_ in terceiros_sorted[:8]]
    slot_map.update(mapear_terceiros(top8, slots_3))

    # --- Quem passou dos grupos ---
    passou_grupos = set(slot_map.values())  # 1º, 2º de 12 grupos + 8 terceiros = 32

    # --- Mata-mata ---
    oitavas = set()
    quartas = set()
    semi = set()
    final_set = set()
    campea = set()

    for row in calendario_df.itertuples(index=False):
        mid = row.match_id  # ex: "M73"
        rnd = row.round

        home = slot_map.get(str(row.home_slot), str(row.home_slot))
        away = slot_map.get(str(row.away_slot), str(row.away_slot))

        lc, lv = obter_lambda(home, away, True, modelos, colunas, elos, lambda_cache)
        gc, gv = simular_jogo(lc, lv)

        if gc > gv:
            vencedor, perdedor = home, away
        elif gv > gc:
            vencedor, perdedor = away, home
        else:
            vencedor, perdedor = (home, away) if np.random.random() < 0.5 else (away, home)

        slot_map[f"W{mid[1:]}"] = vencedor

        if rnd == "R32":
            oitavas.add(vencedor)
        elif rnd == "R16":
            quartas.add(vencedor)
        elif rnd == "QF":
            semi.add(vencedor)
        elif rnd == "SF":
            final_set.add(vencedor)
            slot_map[f"RU{mid[1:]}"] = perdedor
        elif rnd == "Final":
            campea.add(vencedor)

    return {
        "passou_grupos": passou_grupos,
        "oitavas":       oitavas,
        "quartas":       quartas,
        "semi":          semi,
        "final":         final_set,
        "campea":        campea,
    }


def preparar():
    """Carrega todos os dados e pré-computa lambdas. Pensado para @st.cache_resource."""
    copa_df, grupos_equipes, equipe_para_grupo, calendario_df, slots_3, modelos, colunas, elos = carregar_dados()
    lambda_cache = pre_computar_lambdas(copa_df, modelos, colunas, elos)
    return dict(
        copa_df=copa_df,
        grupos_equipes=grupos_equipes,
        equipe_para_grupo=equipe_para_grupo,
        calendario_df=calendario_df,
        slots_3=slots_3,
        lambda_cache=lambda_cache,
        modelos=modelos,
        colunas=colunas,
        elos=elos,
        todas_selecoes=sorted(elos.keys()),
    )


def simular_torneio_detalhado(dados: dict) -> dict:
    """Uma simulação completa com detalhes de grupos e mata-mata para o dashboard."""
    copa_df = dados["copa_df"]
    grupos_equipes = dados["grupos_equipes"]
    equipe_para_grupo = dados["equipe_para_grupo"]
    calendario_df = dados["calendario_df"]
    slots_3 = dados["slots_3"]
    lambda_cache = dados["lambda_cache"]
    modelos = dados["modelos"]
    colunas = dados["colunas"]
    elos = dados["elos"]

    # --- Fase de grupos (tracking estendido) ---
    standings = {
        g: {e: {"pts": 0, "saldo": 0, "gols_pro": 0, "gols_contra": 0, "V": 0, "E": 0, "D": 0}
            for e in equipes}
        for g, equipes in grupos_equipes.items()
    }

    for row in copa_df.itertuples(index=False):
        lc, lv = lambda_cache[(row.time_casa, row.time_visitante, row.neutro, 3)]
        gc, gv = simular_jogo(lc, lv)

        def upd(equipe, gf, gc_opp):
            s = standings[equipe_para_grupo[equipe]][equipe]
            s["gols_pro"] += gf
            s["gols_contra"] += gc_opp
            s["saldo"] += gf - gc_opp
            if gf > gc_opp:
                s["pts"] += 3; s["V"] += 1
            elif gf == gc_opp:
                s["pts"] += 1; s["E"] += 1
            else:
                s["D"] += 1

        upd(row.time_casa, gc, gv)
        upd(row.time_visitante, gv, gc)

    # --- Classificação dos grupos ---
    slot_map = {}
    terceiros = []
    grupos_resultado = {}

    for grupo in sorted(standings.keys()):
        stats = standings[grupo]
        ranking = classificar_grupo(stats)
        slot_map[f"1{grupo}"] = ranking[0][0]
        slot_map[f"2{grupo}"] = ranking[1][0]
        t_equipe, t_stats = ranking[2]
        terceiros.append((t_equipe, grupo, t_stats["pts"], t_stats["saldo"], t_stats["gols_pro"]))

        linhas_grupo = []
        for pos, (equipe, s) in enumerate(ranking, start=1):
            linhas_grupo.append({
                "posicao": pos, "selecao": equipe,
                "jogos": s["V"] + s["E"] + s["D"],
                "vitorias": s["V"], "empates": s["E"], "derrotas": s["D"],
                "gols_pro": s["gols_pro"], "gols_contra": s["gols_contra"],
                "saldo_gols": s["saldo"], "pontos": s["pts"],
            })
        grupos_resultado[grupo] = pd.DataFrame(linhas_grupo)

    # --- 8 melhores terceiros ---
    terceiros_sorted = sorted(terceiros, key=lambda x: (-x[2], -x[3], -x[4], np.random.random()))
    top8 = [(e, g) for e, g, *_ in terceiros_sorted[:8]]
    slot_map.update(mapear_terceiros(top8, slots_3))

    # --- Mata-mata ---
    mata_mata: dict[str, list] = {r: [] for r in NOMES_RODADA}
    campeao = vice = terceiro = None

    for row in calendario_df.itertuples(index=False):
        mid = row.match_id
        rnd = row.round

        home = slot_map.get(str(row.home_slot), str(row.home_slot))
        away = slot_map.get(str(row.away_slot), str(row.away_slot))

        lc, lv = obter_lambda(home, away, True, modelos, colunas, elos, lambda_cache)
        gc, gv = simular_jogo(lc, lv)
        penaltis = (gc == gv)

        if gc > gv:
            vencedor, perdedor = home, away
        elif gv > gc:
            vencedor, perdedor = away, home
        else:
            vencedor, perdedor = (home, away) if np.random.random() < 0.5 else (away, home)

        slot_map[f"W{mid[1:]}"] = vencedor
        if rnd == "SF":
            slot_map[f"RU{mid[1:]}"] = perdedor

        mata_mata[rnd].append({
            "home": home, "away": away,
            "gols_home": gc, "gols_away": gv,
            "vencedor": vencedor, "penaltis": penaltis,
        })

        if rnd == "Final":
            campeao, vice = vencedor, perdedor
        elif rnd == "3rd":
            terceiro = vencedor

    return {
        "campeao": campeao,
        "vice": vice,
        "terceiro": terceiro,
        "mata_mata": mata_mata,
        "grupos": grupos_resultado,
    }


def gravar(df):
    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS gold_probabilidades_copa")
            cur.execute(CREATE_SQL)
            buf = io.StringIO()
            df.to_csv(buf, index=False, na_rep="")
            buf.seek(0)
            cur.copy_expert(COPY_SQL, buf)
        conn.commit()
        print(f"Tabela gold_probabilidades_copa gravada com {len(df)} linhas.")
    finally:
        conn.close()


if __name__ == "__main__":
    print("Carregando dados...")
    copa_df, grupos_equipes, equipe_para_grupo, calendario_df, slots_3, modelos, colunas, elos = carregar_dados()

    print("Pre-computando lambdas dos 72 jogos de grupo...")
    lambda_cache = pre_computar_lambdas(copa_df, modelos, colunas, elos)

    print(f"Simulando {N_SIMS} vezes (seed={SEED})...")
    np.random.seed(SEED)
    contagens = {fase: defaultdict(int) for fase in FASES}

    for i in range(N_SIMS):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N_SIMS}")
        res = uma_simulacao(
            copa_df, grupos_equipes, equipe_para_grupo,
            calendario_df, slots_3, lambda_cache, modelos, colunas, elos,
        )
        for fase, equipes in res.items():
            for e in equipes:
                contagens[fase][e] += 1

    # --- Montar DataFrame ---
    todas_equipes = sorted({e for c in contagens.values() for e in c})
    linhas = []
    for selecao in todas_equipes:
        linhas.append({
            "selecao":      selecao,
            "prob_grupo":   contagens["passou_grupos"][selecao] / N_SIMS,
            "prob_oitavas": contagens["oitavas"][selecao] / N_SIMS,
            "prob_quartas": contagens["quartas"][selecao] / N_SIMS,
            "prob_semi":    contagens["semi"][selecao] / N_SIMS,
            "prob_final":   contagens["final"][selecao] / N_SIMS,
            "prob_campea":  contagens["campea"][selecao] / N_SIMS,
        })

    df = pd.DataFrame(linhas).sort_values("prob_campea", ascending=False)

    print(f"\n{'='*55}")
    print("Top 10 — prob_campea")
    print(f"{'='*55}")
    for _, r in df.head(10).iterrows():
        print(f"  {r['selecao']:<30} {r['prob_campea']*100:.1f}%")
    print(f"\nSoma prob_campea: {df['prob_campea'].sum()*100:.1f}%")
    print(f"Soma prob_grupo : {df['prob_grupo'].sum():.1f}")
    print(f"{'='*55}\n")

    gravar(df)
