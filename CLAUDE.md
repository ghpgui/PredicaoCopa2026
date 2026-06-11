# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projeto

**IAPredict** — pipeline de ML que prevê a Copa 2026 via regressão Poisson + ELO + Monte Carlo.
A previsão compete num bolão contra humanos.

Spec completa em `llm/prd.md`. Para cada feature: ler `llm/feature_NN.md` → implementar → rodar a Verificação (SQL) descrita no arquivo.

## Comandos

```bash
# Rodar uma feature individualmente (src/ entra no sys.path automaticamente)
python src/bronze.py
python src/silver.py
# ... etc.

# Dashboard
streamlit run app.py

# Dependências
pip install -r requirements.txt
```

## Arquitetura

### Estrutura de diretórios esperada

```
├── data/
│   ├── results.csv             # jogos internacionais 1872→2026 (inclui 72 jogos Copa 2026 sem placar)
│   ├── calendario_copa2026.csv # mata-mata M73–M104 (referência fixa)
│   └── grupos_copa2026.csv     # grupos A–L, 48 seleções (referência fixa)
├── src/
│   ├── db.py          # get_engine() + get_raw_connection() — lê DATABASE_URL via dotenv
│   ├── bronze.py      # feature_01: ingestão → bronze_jogos
│   ├── silver.py      # feature_02: limpeza + anti-leakage → silver_jogos, silver_copa2026
│   ├── pesos.py       # feature_03: peso_torneio + peso_recencia → silver_ponderado
│   ├── elo.py         # feature_04: ELO → silver_elo_pre_jogo, silver_elo_atual
│   ├── gold.py        # feature_05: atributos de treino → gold_atributos
│   ├── treino.py      # feature_06: GLM Poisson + validação → .pkl + metricas_validacao
│   ├── previsao.py    # feature_07: prever_jogo + experimentos → previsoes, experimentos_mae
│   ├── monte_carlo.py # feature_08: simulação → gold_probabilidades_copa
│   ├── poisson.py     # utilitário compartilhado: prob V/E/D a partir de dois λ (MAX_GOLS=10)
│   └── bandeiras.py   # nome da seleção → emoji de bandeira
├── models/            # modelo_poisson_casa.pkl, modelo_poisson_visitante.pkl, colunas_atributos.pkl
├── app.py             # Streamlit (3 páginas)
├── .env               # não versionar — contém DATABASE_URL
└── .env.example       # versionar
```

`src/` **não é um pacote Python** — imports entre módulos são flat (`from db import get_engine`).
O script insere `src/` no `sys.path` ao rodar; `app.py` faz `sys.path.insert(0, "src")`.

### Camadas do banco (medallion)

| Prefixo | Tabelas |
|---------|---------|
| Bronze | `bronze_jogos` |
| Silver | `silver_jogos`, `silver_copa2026`, `silver_ponderado`, `silver_elo_pre_jogo`, `silver_elo_atual` |
| Gold | `gold_atributos`, `gold_probabilidades_copa` |
| Sem prefixo (saída de modelo) | `metricas_validacao`, `previsoes`, `experimentos_mae` |

### Padrão de escrita no banco

Toda tabela é **idempotente**: `DROP TABLE IF EXISTS` + `CREATE TABLE` com `id bigint generated always as identity primary key` + carga via `COPY ... FROM STDIN` (buffer CSV em memória, `NULL ''`). Nunca usar `INSERT` linha a linha.

## Regras críticas

### Anti data leakage (inegociável)
O modelo treina **somente em dados históricos**. Os 72 jogos da Copa 2026 — identificados por `gols_casa IS NULL` no `results.csv` — **nunca entram no treino**. Eles vão para `silver_copa2026`, não `silver_jogos`.

### Idioma
- **Estrutura** (nomes de colunas, tabelas, variáveis, arquivos): **português**, snake_case.
- **Valores de dados** (`torneio`, `time_casa`, `time_visitante`): manter **inglês original** do CSV. Não traduzir — quebra joins e padronização de nomes.

### Dicionário de colunas (`results.csv` → banco)

| CSV original | Coluna no banco |
|---|---|
| `date` | `data` |
| `home_team` | `time_casa` |
| `away_team` | `time_visitante` |
| `home_score` | `gols_casa` |
| `away_score` | `gols_visitante` |
| `tournament` | `torneio` |
| `city` | `cidade` |
| `country` | `pais` |
| `neutral` | `neutro` |

## Parâmetros canônicos (não alterar)

- **Janela temporal**: apenas `data >= 2006-01-01`
- **DATA_REF** (recência): `2026-06-11`
- **`peso_recencia`**: `0.5 ** (idade_anos / 5)`, `idade_anos = (DATA_REF - data).days / 365.25`
- **`peso_torneio`**: Nível 3 = {`FIFA World Cup`, `Confederations Cup`, `CONMEBOL–UEFA Cup of Champions`}; Nível 2 = contém `qualification` ou `nations league` (case-insensitive) + {`UEFA Euro`, `Copa América`, `African Cup of Nations`, `AFC Asian Cup`, `Gold Cup`, `Oceania Nations Cup`}; Nível 1 = tudo mais
- **ELO**: início 1500; HFA = 100 (quando `neutro=False`); K = 20/40/60 por nível 1/2/3; ELO gravado é o **pré-jogo**
- **Treino Poisson**: 6 atributos (`elo_casa, elo_visitante, dif_elo, neutro, peso_torneio, peso_recencia`); `add_constant`; `var_weights = peso_torneio × peso_recencia`; split temporal: treino `< 2024-01-01`, teste `>= 2024-01-01`; excluir amistosos do treino
- **Monte Carlo**: `N=1000`, `seed=42`; mata-mata em sede neutra; empate → 50/50; melhores terceiros via `scipy.optimize.linear_sum_assignment`

## Conexão ao banco

Usar **connection string direta** (SQLAlchemy/psycopg2 via `DATABASE_URL`), não MCP, para carga em massa.

```python
# .env.example
DATABASE_URL=postgresql://usuario:senha@host:5432/iapredict
CAMINHO_CSV=data/results.csv
```
