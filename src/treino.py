import io
import os
import pickle
import sys
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from db import get_raw_connection, read_sql
from poisson import resultado_previsto, resultado_real

load_dotenv()

COLUNAS_ATRIBUTOS = [
    "elo_casa", "elo_visitante", "dif_elo", "neutro",
    "peso_torneio", "peso_recencia",
]

DATA_SPLIT = date(2024, 1, 1)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CREATE_SQL = """
CREATE TABLE metricas_validacao (
    id            bigint generated always as identity primary key,
    mae_casa      double precision,
    mae_visitante double precision,
    acuracia      double precision
)
"""

COPY_SQL = """
COPY metricas_validacao (mae_casa, mae_visitante, acuracia)
FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
"""


def preparar(df: pd.DataFrame):
    df["neutro"] = df["neutro"].astype(int)
    mask_treino = df["data"].apply(lambda d: d < DATA_SPLIT)
    mask_teste = ~mask_treino
    return df[mask_treino].copy(), df[mask_teste].copy()


def montar_X(df: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(df[COLUNAS_ATRIBUTOS].astype(float), has_constant="add")


def treinar(df_treino: pd.DataFrame):
    X = montar_X(df_treino)
    peso = (df_treino["peso_torneio"] * df_treino["peso_recencia"]).values

    modelo_casa = sm.GLM(
        df_treino["gols_casa"].astype(float), X,
        family=sm.families.Poisson(), var_weights=peso,
    ).fit()

    modelo_visit = sm.GLM(
        df_treino["gols_visitante"].astype(float), X,
        family=sm.families.Poisson(), var_weights=peso,
    ).fit()

    return modelo_casa, modelo_visit


def validar(df_teste: pd.DataFrame, modelo_casa, modelo_visit):
    X_teste = montar_X(df_teste)
    lambda_casa = modelo_casa.predict(X_teste).values
    lambda_visit = modelo_visit.predict(X_teste).values

    mae_casa = float(np.abs(lambda_casa - df_teste["gols_casa"].values).mean())
    mae_visit = float(np.abs(lambda_visit - df_teste["gols_visitante"].values).mean())

    acertos = sum(
        resultado_previsto(lc, lv) == resultado_real(gc, gv)
        for lc, lv, gc, gv in zip(
            lambda_casa, lambda_visit,
            df_teste["gols_casa"].values,
            df_teste["gols_visitante"].values,
        )
    )
    acuracia = acertos / len(df_teste)
    return mae_casa, mae_visit, acuracia


def salvar_modelos(modelo_casa, modelo_visit):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for nome, obj in [
        ("modelo_poisson_casa.pkl", modelo_casa),
        ("modelo_poisson_visitante.pkl", modelo_visit),
        ("colunas_atributos.pkl", COLUNAS_ATRIBUTOS),
    ]:
        with open(os.path.join(MODELS_DIR, nome), "wb") as f:
            pickle.dump(obj, f)
        print(f"Salvo: models/{nome}")


def gravar_metricas(mae_casa, mae_visit, acuracia):
    df = pd.DataFrame([{
        "mae_casa": mae_casa,
        "mae_visitante": mae_visit,
        "acuracia": acuracia,
    }])
    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS metricas_validacao")
            cur.execute(CREATE_SQL)
            buf = io.StringIO()
            df.to_csv(buf, index=False, na_rep="")
            buf.seek(0)
            cur.copy_expert(COPY_SQL, buf)
        conn.commit()
        print("Tabela metricas_validacao gravada.")
    finally:
        conn.close()


def imprimir_metricas(mae_casa, mae_visit, acuracia, n_treino, n_teste):
    print(f"\n{'='*50}")
    print("Validação do modelo")
    print(f"{'='*50}")
    print(f"Treino : {n_treino:,} jogos (até 2023)")
    print(f"Teste  : {n_teste:,} jogos (2024+)")
    print(f"MAE casa     : {mae_casa:.4f}")
    print(f"MAE visitante: {mae_visit:.4f}")
    print(f"Acurácia     : {acuracia:.4f}  ({acuracia*100:.1f}%)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    df = read_sql("SELECT * FROM gold_atributos ORDER BY data, jogo_id")
    df_treino, df_teste = preparar(df)

    modelo_casa, modelo_visit = treinar(df_treino)
    mae_casa, mae_visit, acuracia = validar(df_teste, modelo_casa, modelo_visit)

    imprimir_metricas(mae_casa, mae_visit, acuracia, len(df_treino), len(df_teste))
    salvar_modelos(modelo_casa, modelo_visit)
    gravar_metricas(mae_casa, mae_visit, acuracia)
