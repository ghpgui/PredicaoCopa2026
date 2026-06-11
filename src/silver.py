import io
import os
import sys
from datetime import date

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from db import get_raw_connection, read_sql

load_dotenv()

NOMES = {
    # "Nome variante": "Nome canônico",
}

CREATE_SQL = """
CREATE TABLE {tabela} (
    id             bigint generated always as identity primary key,
    data           date,
    time_casa      text,
    time_visitante text,
    gols_casa      integer,
    gols_visitante integer,
    torneio        text,
    cidade         text,
    pais           text,
    neutro         boolean,
    eh_amistoso    boolean
)
"""

COPY_SQL = """
COPY {tabela} (data, time_casa, time_visitante, gols_casa, gols_visitante,
               torneio, cidade, pais, neutro, eh_amistoso)
FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
"""

COLUNAS_NEGOCIO = [
    "data", "time_casa", "time_visitante", "gols_casa", "gols_visitante",
    "torneio", "cidade", "pais", "neutro",
]


def carregar_bronze() -> pd.DataFrame:
    df = read_sql("SELECT * FROM bronze_jogos ORDER BY id")
    df["gols_casa"] = df["gols_casa"].astype("Int64")
    df["gols_visitante"] = df["gols_visitante"].astype("Int64")
    return df


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    df["time_casa"] = df["time_casa"].str.strip().replace(NOMES)
    df["time_visitante"] = df["time_visitante"].str.strip().replace(NOMES)
    df = df.drop_duplicates(subset=COLUNAS_NEGOCIO)
    df["eh_amistoso"] = df["torneio"] == "Friendly"
    return df


def split(df: pd.DataFrame):
    mascara_copa = df["gols_casa"].isna()
    df_copa = df[mascara_copa].copy()
    df_hist = df[~mascara_copa & (df["data"] >= date(2006, 1, 1))].copy()
    return df_hist, df_copa


def imprimir_inventario(df_hist: pd.DataFrame, df_copa: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print("silver — inventário")
    print(f"{'='*55}")
    print(f"silver_jogos  : {len(df_hist):>7,} linhas")
    print(f"  amistosos   : {df_hist['eh_amistoso'].sum():>7,}")
    print(f"  competitivos: {(~df_hist['eh_amistoso']).sum():>7,}")
    print(f"silver_copa2026: {len(df_copa):>6,} linhas")
    print(f"{'='*55}\n")


def gravar(df: pd.DataFrame, tabela: str) -> None:
    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {tabela}")
            cur.execute(CREATE_SQL.format(tabela=tabela))

            colunas = [
                "data", "time_casa", "time_visitante", "gols_casa", "gols_visitante",
                "torneio", "cidade", "pais", "neutro", "eh_amistoso",
            ]
            buf = io.StringIO()
            df[colunas].to_csv(buf, index=False, na_rep="")
            buf.seek(0)
            cur.copy_expert(COPY_SQL.format(tabela=tabela), buf)

        conn.commit()
        print(f"Tabela {tabela} gravada com {len(df):,} linhas.")
    finally:
        conn.close()


if __name__ == "__main__":
    df = carregar_bronze()
    df = limpar(df)
    df_hist, df_copa = split(df)
    imprimir_inventario(df_hist, df_copa)
    gravar(df_hist, "silver_jogos")
    gravar(df_copa, "silver_copa2026")
