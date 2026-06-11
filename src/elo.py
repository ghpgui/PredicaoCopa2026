import io
import os
import sys
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from db import get_raw_connection, read_sql

load_dotenv()

K_POR_NIVEL = {1: 20, 2: 40, 3: 60}

CREATE_PRE_JOGO = """
CREATE TABLE silver_elo_pre_jogo (
    id             bigint generated always as identity primary key,
    jogo_id        bigint,
    data           date,
    time_casa      text,
    time_visitante text,
    elo_casa       double precision,
    elo_visitante  double precision
)
"""

CREATE_ATUAL = """
CREATE TABLE silver_elo_atual (
    id      bigint generated always as identity primary key,
    selecao text,
    elo     double precision
)
"""

COPY_PRE_JOGO = """
COPY silver_elo_pre_jogo (jogo_id, data, time_casa, time_visitante, elo_casa, elo_visitante)
FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
"""

COPY_ATUAL = """
COPY silver_elo_atual (selecao, elo)
FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
"""


def calcular_elo(df: pd.DataFrame):
    elo = defaultdict(lambda: 1500.0)
    linhas_pre = []

    for row in df.itertuples(index=False):
        casa = row.time_casa
        visit = row.time_visitante

        elo_casa_pre = elo[casa]
        elo_visit_pre = elo[visit]

        linhas_pre.append((
            row.id, row.data, casa, visit, elo_casa_pre, elo_visit_pre,
        ))

        hfa = 0.0 if row.neutro else 100.0
        e_casa = 1.0 / (1.0 + 10.0 ** ((elo_visit_pre - elo_casa_pre - hfa) / 400.0))
        e_visit = 1.0 - e_casa

        if row.gols_casa > row.gols_visitante:
            s_casa = 1.0
        elif row.gols_casa == row.gols_visitante:
            s_casa = 0.5
        else:
            s_casa = 0.0

        k = K_POR_NIVEL[row.peso_torneio]
        elo[casa] += k * (s_casa - e_casa)
        elo[visit] += k * ((1.0 - s_casa) - e_visit)

    df_pre = pd.DataFrame(
        linhas_pre,
        columns=["jogo_id", "data", "time_casa", "time_visitante", "elo_casa", "elo_visitante"],
    )
    df_atual = pd.DataFrame(
        sorted(elo.items(), key=lambda x: -x[1]),
        columns=["selecao", "elo"],
    )
    return df_pre, df_atual


def imprimir_inventario(df_pre: pd.DataFrame, df_atual: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print("ELO — inventário")
    print(f"{'='*55}")
    print(f"silver_elo_pre_jogo: {len(df_pre):,} linhas")
    print(f"silver_elo_atual   : {len(df_atual):,} seleções")
    print(f"\nTop 10 ELO atual:")
    for _, r in df_atual.head(10).iterrows():
        print(f"  {r['selecao']:<30} {r['elo']:.1f}")
    print(f"{'='*55}\n")


def gravar(df: pd.DataFrame, tabela: str, create_sql: str, copy_sql: str) -> None:
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
    df = read_sql("SELECT * FROM silver_ponderado ORDER BY data, id")
    df_pre, df_atual = calcular_elo(df)
    imprimir_inventario(df_pre, df_atual)
    gravar(df_pre, "silver_elo_pre_jogo", CREATE_PRE_JOGO, COPY_PRE_JOGO)
    gravar(df_atual, "silver_elo_atual", CREATE_ATUAL, COPY_ATUAL)
