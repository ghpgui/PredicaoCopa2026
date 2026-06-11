import io
import os
import sys
from datetime import date

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from db import get_raw_connection, read_sql

load_dotenv()

DATA_REF = date(2026, 6, 11)

NIVEL3 = {"FIFA World Cup", "Confederations Cup", "CONMEBOL–UEFA Cup of Champions"}
CONTINENTAIS = {
    "UEFA Euro", "Copa América", "African Cup of Nations",
    "AFC Asian Cup", "Gold Cup", "Oceania Nations Cup",
}

CREATE_SQL = """
CREATE TABLE silver_ponderado (
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
    eh_amistoso    boolean,
    peso_torneio   integer,
    peso_recencia  double precision
)
"""

COPY_SQL = """
COPY silver_ponderado (data, time_casa, time_visitante, gols_casa, gols_visitante,
                       torneio, cidade, pais, neutro, eh_amistoso,
                       peso_torneio, peso_recencia)
FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
"""

COLUNAS = [
    "data", "time_casa", "time_visitante", "gols_casa", "gols_visitante",
    "torneio", "cidade", "pais", "neutro", "eh_amistoso",
    "peso_torneio", "peso_recencia",
]


def classificar(torneio: str) -> int:
    if torneio in NIVEL3:
        return 3
    t = torneio.lower()
    if "qualification" in t or "nations league" in t or torneio in CONTINENTAIS:
        return 2
    return 1


def calcular_pesos(df: pd.DataFrame) -> pd.DataFrame:
    df["peso_torneio"] = df["torneio"].apply(classificar)
    df["idade_anos"] = df["data"].apply(lambda d: (DATA_REF - d).days / 365.25)
    df["peso_recencia"] = 0.5 ** (df["idade_anos"] / 5)
    df = df.drop(columns=["idade_anos"])
    return df


def imprimir_inventario(df: pd.DataFrame) -> None:
    dist = df["peso_torneio"].value_counts().sort_index()
    print(f"\n{'='*50}")
    print("silver_ponderado — inventário")
    print(f"{'='*50}")
    print(f"Total: {len(df):,} linhas")
    print("\npeso_torneio:")
    for nivel, cnt in dist.items():
        print(f"  nível {nivel}: {cnt:>6,}")
    print(f"\npeso_recencia: min={df['peso_recencia'].min():.6f}  max={df['peso_recencia'].max():.6f}")
    print(f"{'='*50}\n")


def gravar(df: pd.DataFrame) -> None:
    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS silver_ponderado")
            cur.execute(CREATE_SQL)

            buf = io.StringIO()
            df[COLUNAS].to_csv(buf, index=False, na_rep="")
            buf.seek(0)
            cur.copy_expert(COPY_SQL, buf)

        conn.commit()
        print(f"Tabela silver_ponderado gravada com {len(df):,} linhas.")
    finally:
        conn.close()


if __name__ == "__main__":
    df = read_sql("SELECT * FROM silver_jogos ORDER BY id")
    df = calcular_pesos(df)
    imprimir_inventario(df)
    gravar(df)
