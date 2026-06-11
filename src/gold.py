import io
import os
import sys

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from db import get_raw_connection, read_sql

load_dotenv()

QUERY = """
SELECT
    s.id        AS jogo_id,
    s.data,
    s.time_casa,
    s.time_visitante,
    e.elo_casa,
    e.elo_visitante,
    s.neutro,
    s.peso_torneio,
    s.peso_recencia,
    s.gols_casa,
    s.gols_visitante
FROM silver_ponderado s
JOIN silver_elo_pre_jogo e ON e.jogo_id = s.id
WHERE NOT s.eh_amistoso
ORDER BY s.data, s.id
"""

ATRIBUTOS = [
    "elo_casa", "elo_visitante", "dif_elo", "neutro",
    "peso_torneio", "peso_recencia", "gols_casa", "gols_visitante",
]

COLUNAS = [
    "jogo_id", "data", "time_casa", "time_visitante",
    "elo_casa", "elo_visitante", "dif_elo", "neutro",
    "peso_torneio", "peso_recencia", "gols_casa", "gols_visitante",
]

CREATE_SQL = """
CREATE TABLE gold_atributos (
    id             bigint generated always as identity primary key,
    jogo_id        bigint,
    data           date,
    time_casa      text,
    time_visitante text,
    elo_casa       double precision,
    elo_visitante  double precision,
    dif_elo        double precision,
    neutro         boolean,
    peso_torneio   integer,
    peso_recencia  double precision,
    gols_casa      integer,
    gols_visitante integer
)
"""

COPY_SQL = """
COPY gold_atributos (jogo_id, data, time_casa, time_visitante,
                     elo_casa, elo_visitante, dif_elo, neutro,
                     peso_torneio, peso_recencia, gols_casa, gols_visitante)
FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
"""


def montar(df: pd.DataFrame) -> pd.DataFrame:
    df["dif_elo"] = df["elo_casa"] - df["elo_visitante"]
    nulos = df[ATRIBUTOS].isna().sum().sum()
    assert nulos == 0, f"Nulos inesperados nos atributos: {nulos}"
    return df


def imprimir_inventario(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print("gold_atributos — inventário")
    print(f"{'='*55}")
    print(f"Total: {len(df):,} linhas")
    print("\nPrimeiras 3 linhas:")
    print(df[COLUNAS].head(3).to_string(index=False))
    print(f"{'='*55}\n")


def gravar(df: pd.DataFrame) -> None:
    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS gold_atributos")
            cur.execute(CREATE_SQL)
            buf = io.StringIO()
            df[COLUNAS].to_csv(buf, index=False, na_rep="")
            buf.seek(0)
            cur.copy_expert(COPY_SQL, buf)
        conn.commit()
        print(f"Tabela gold_atributos gravada com {len(df):,} linhas.")
    finally:
        conn.close()


if __name__ == "__main__":
    df = read_sql(QUERY)
    df = montar(df)
    imprimir_inventario(df)
    gravar(df)
