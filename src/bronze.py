import io
import os
import sys

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from db import get_raw_connection

load_dotenv()

COLUNAS = {
    "date": "data",
    "home_team": "time_casa",
    "away_team": "time_visitante",
    "home_score": "gols_casa",
    "away_score": "gols_visitante",
    "tournament": "torneio",
    "city": "cidade",
    "country": "pais",
    "neutral": "neutro",
}

CREATE_SQL = """
CREATE TABLE bronze_jogos (
    id            bigint generated always as identity primary key,
    data          date,
    time_casa     text,
    time_visitante text,
    gols_casa     integer,
    gols_visitante integer,
    torneio       text,
    cidade        text,
    pais          text,
    neutro        boolean
)
"""

COPY_SQL = """
COPY bronze_jogos (data, time_casa, time_visitante, gols_casa, gols_visitante,
                   torneio, cidade, pais, neutro)
FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
"""


def carregar_csv(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=["date"])
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    df["neutral"] = df["neutral"].astype(bool)
    df = df.rename(columns=COLUNAS)
    return df


def imprimir_inventario(df: pd.DataFrame) -> None:
    print(f"\n{'='*50}")
    print(f"bronze_jogos — inventário")
    print(f"{'='*50}")
    print(f"Linhas: {len(df):,}")
    print(f"\nTipos:")
    for col, dtype in df.dtypes.items():
        nulos = df[col].isna().sum()
        pct = nulos / len(df) * 100
        print(f"  {col:<20} {str(dtype):<12} nulos: {nulos:>6} ({pct:.1f}%)")
    print(f"{'='*50}\n")


def gravar_bronze(df: pd.DataFrame) -> None:
    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS bronze_jogos")
            cur.execute(CREATE_SQL)

            buf = io.StringIO()
            df.to_csv(buf, index=False, na_rep="")
            buf.seek(0)
            cur.copy_expert(COPY_SQL, buf)

        conn.commit()
        print(f"Tabela bronze_jogos gravada com {len(df):,} linhas.")
    finally:
        conn.close()


if __name__ == "__main__":
    caminho = sys.argv[1] if len(sys.argv) > 1 else os.getenv("CAMINHO_CSV", "data/results.csv")
    df = carregar_csv(caminho)
    imprimir_inventario(df)
    gravar_bronze(df)
