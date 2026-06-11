import io
import os
import pickle
import sys
from datetime import date

import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from db import get_raw_connection, read_sql
from poisson import probabilidades_resultado
from treino import DATA_SPLIT, treinar, validar

load_dotenv()

PESO_TORNEIO_COPA = 3
DATA_REF = date(2026, 6, 11)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CREATE_PREVISOES = """
CREATE TABLE previsoes (
    id                      bigint generated always as identity primary key,
    time_casa               text,
    time_visitante          text,
    gols_esperados_casa     double precision,
    gols_esperados_visitante double precision,
    prob_vitoria            double precision,
    prob_empate             double precision,
    prob_derrota            double precision
)
"""

COPY_PREVISOES = """
COPY previsoes (time_casa, time_visitante, gols_esperados_casa, gols_esperados_visitante,
                prob_vitoria, prob_empate, prob_derrota)
FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
"""

CREATE_EXPERIMENTOS = """
CREATE TABLE experimentos_mae (
    id            bigint generated always as identity primary key,
    config        text,
    mae_casa      double precision,
    mae_visitante double precision
)
"""

COPY_EXPERIMENTOS = """
COPY experimentos_mae (config, mae_casa, mae_visitante)
FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
"""

CONFIGS_RECENCIA = {
    "sem_recencia": lambda idade: 1.0,
    "meia_vida_3":  lambda idade: 0.5 ** (idade / 3),
    "meia_vida_5":  lambda idade: 0.5 ** (idade / 5),
    "meia_vida_10": lambda idade: 0.5 ** (idade / 10),
}


def carregar_artefatos():
    def pkl(nome):
        return pickle.load(open(os.path.join(MODELS_DIR, nome), "rb"))

    modelos = {"casa": pkl("modelo_poisson_casa.pkl"), "visit": pkl("modelo_poisson_visitante.pkl")}
    colunas = pkl("colunas_atributos.pkl")
    df_elo = read_sql("SELECT selecao, elo FROM silver_elo_atual")
    elos = dict(zip(df_elo["selecao"], df_elo["elo"]))
    return modelos, colunas, elos


carregar_modelos = carregar_artefatos


def prever_jogo(time_casa, time_visitante, neutro, peso_torneio, modelos, colunas, elos):
    elo_casa = elos.get(time_casa, 1500.0)
    elo_visit = elos.get(time_visitante, 1500.0)
    features = pd.DataFrame([{
        "elo_casa": elo_casa, "elo_visitante": elo_visit,
        "dif_elo": elo_casa - elo_visit,
        "neutro": int(neutro), "peso_torneio": peso_torneio,
        "peso_recencia": 1.0,
    }])[colunas]
    X = sm.add_constant(features.astype(float), has_constant="add")
    lc = float(modelos["casa"].predict(X).iloc[0])
    lv = float(modelos["visit"].predict(X).iloc[0])
    pv, pe, pd_ = probabilidades_resultado(lc, lv)
    return {
        "gols_esperados_casa": lc, "gols_esperados_visitante": lv,
        "prob_vitoria": pv, "prob_empate": pe, "prob_derrota": pd_,
    }


def gerar_previsoes(modelos, colunas, elos):
    df_copa = read_sql("SELECT time_casa, time_visitante, neutro FROM silver_copa2026")
    linhas = []
    for row in df_copa.itertuples(index=False):
        r = prever_jogo(row.time_casa, row.time_visitante, row.neutro, 3, modelos, colunas, elos)
        linhas.append({"time_casa": row.time_casa, "time_visitante": row.time_visitante, **r})
    return pd.DataFrame(linhas)


def rodar_experimentos():
    df = read_sql("SELECT * FROM gold_atributos ORDER BY data, jogo_id")
    df["neutro"] = df["neutro"].astype(int)
    df["idade_anos"] = df["data"].apply(lambda d: (DATA_REF - d).days / 365.25)

    mask_treino = df["data"].apply(lambda d: d < DATA_SPLIT)
    resultados = []

    for config, fn_recencia in CONFIGS_RECENCIA.items():
        df_mod = df.copy()
        df_mod["peso_recencia"] = df_mod["idade_anos"].apply(fn_recencia)

        df_treino = df_mod[mask_treino].copy()
        df_teste = df_mod[~mask_treino].copy()

        mod_casa, mod_visit = treinar(df_treino)
        mae_casa, mae_visit, _ = validar(df_teste, mod_casa, mod_visit)

        resultados.append({"config": config, "mae_casa": mae_casa, "mae_visitante": mae_visit})
        print(f"  {config:<15} MAE casa={mae_casa:.4f}  MAE visit={mae_visit:.4f}")

    return pd.DataFrame(resultados)


def gravar(df, tabela, create_sql, copy_sql):
    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {tabela}")
            cur.execute(create_sql)
            buf = io.StringIO()
            df.to_csv(buf, index=False, na_rep="")
            buf.seek(0)
            cur.copy_expert(copy_sql, buf)
        conn.commit()
        print(f"Tabela {tabela} gravada com {len(df):,} linhas.")
    finally:
        conn.close()


if __name__ == "__main__":
    modelos, colunas, elos = carregar_artefatos()

    print("Gerando previsoes para 72 jogos da Copa 2026...")
    df_prev = gerar_previsoes(modelos, colunas, elos)

    print(f"\nExperimentos de recência (re-treino completo):")
    df_exp = rodar_experimentos()

    print(f"\n{'='*55}")
    print(f"previsoes      : {len(df_prev)} linhas")
    print(f"experimentos_mae: {len(df_exp)} configs")
    print(f"{'='*55}\n")

    gravar(df_prev, "previsoes", CREATE_PREVISOES, COPY_PREVISOES)
    gravar(df_exp, "experimentos_mae", CREATE_EXPERIMENTOS, COPY_EXPERIMENTOS)
