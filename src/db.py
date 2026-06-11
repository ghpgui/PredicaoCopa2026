import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_DATABASE_URL = os.environ["DATABASE_URL"]


def get_engine():
    return create_engine(_DATABASE_URL)


def get_raw_connection():
    return psycopg2.connect(_DATABASE_URL)


def read_sql(query: str, params=None) -> pd.DataFrame:
    """Lê query SQL e retorna DataFrame via psycopg2."""
    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()
