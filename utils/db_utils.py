import pandas as pd
import duckdb
from contextlib import contextmanager
import numpy as np
from typing import List, Union
from contextlib import closing
from collections.abc import Iterable
from functools import reduce

#############################################################
#PIPELINES TO GRAB SPECIFIC DATA#############################
#These functions are small and quick to grab specific data#
#############################################################
def state_ids_from_counties(con, county_FIPs: list[str]) -> list[str]:
    """
    Given a list of county FIPS codes (possibly with leading zeros),
    return the distinct state_id values for those counties.
    Handles mismatched leading zeros between county_FIPs and database.
    """
    if not county_FIPs:
        return []

    query = """
        WITH nf AS (
            -- Normalize county IDs (strip leading zeros like '06037' -> '6037')
            SELECT REGEXP_REPLACE(CAST(x AS VARCHAR), '^0+', '') AS county_id_norm
            FROM (SELECT UNNEST(?) AS x)
        ),
        nc AS (
            -- Normalize county table IDs too
            SELECT 
                REGEXP_REPLACE(CAST(c.County_ID AS VARCHAR), '^0+', '') AS county_id_norm,
                REGEXP_REPLACE(CAST(c.state_ID AS VARCHAR), '^0+', '') AS state_id_norm
            FROM raw.counties c
        )
        SELECT DISTINCT LPAD(nc.state_id_norm, 2, '0') AS state_id
        FROM nc
        JOIN nf ON nc.county_id_norm = nf.county_id_norm
        ORDER BY state_id;
    """

    df = con.execute(query, [county_FIPs]).df()
    return df["state_id"].tolist()


def get_question_ids(con, question_names: list[str]) -> list[str]:
    # normalize names to lowercase and trim
    question_names = [q.strip().lower() for q in question_names if q.strip()]
    if not question_names:
        return []

    query = """
        SELECT question_id, question
        FROM raw.questions
        WHERE LOWER(TRIM(question)) IN (SELECT UNNEST(?))
        ORDER BY question_id
    """

    df = con.execute(query, [question_names]).df()
    return df["question_id"].tolist()

def get_all_variable(con, variable_name: str, table_name: str) -> pd.DataFrame:
    if isinstance(variable_name, str):
        query = f"""
        SELECT {variable_name}
        FROM {table_name}
        """
        rows = con.execute(query).fetchall()
        row = list(rows)
        return [value[0] for value in row]
    elif isinstance(variable_name, (list, tuple)):
        columns = ", ".join(variable_name)
        query = f"""
        SELECT {columns}
        FROM {table_name}
        """
        df = con.execute(query).df()
        return df
    else:
        raise TypeError("variable_name must be a string or list of strings")

def get_columns_in_table(con, table_name: str) -> pd.DataFrame:
    query = f"""
    PRAGMA table_info('{table_name}')
    """
    rows = con.execute(query).fetchall()
    row = list(rows)
    return [value[1] for value in row if value[1] != 'state_id']

def grab_county_names(con: duckdb.DuckDBPyConnection,
                      fips_list: Union[str, List[Union[str, int]]]) -> List[str]:
    if isinstance(fips_list, str):
        items = [x.strip().lstrip("0") for x in fips_list.split(",") if x.strip()]
    elif isinstance(fips_list, list):
        items = [str(x).strip().lstrip("0") for x in fips_list if str(x).strip()]
    else:
        raise TypeError("fips_list must be a string or list of FIPS codes.")

    if not items:
        return []

    query = """
        SELECT name
        FROM raw.counties
        WHERE County_id IN (SELECT * FROM UNNEST(?))
        ORDER BY name
    """
    df: pd.DataFrame = con.execute(query, [items]).df()
    return df["name"].tolist()


def grab_state_ids(state_abbrev, con):
    if isinstance(state_abbrev, str):
        items = [s.strip().upper() for s in state_abbrev.split(",") if s.strip()]
    elif isinstance(state_abbrev, list):
        items = [str(s).strip().upper() for s in state_abbrev if str(s).strip()]
    query = """
        SELECT state_id
        FROM raw.states
        WHERE abbrev IN (SELECT * FROM UNNEST(?))
        ORDER BY state_id
    """
    df = con.execute(query, [items]).df()
    return df["state_id"].tolist()

def grab_census_variable_ids(variable_names: list[str], con) -> dict[str, str]:
    # normalize names to lowercase and trim
    variable_names = [v.strip().lower() for v in variable_names if v.strip()]
    if not variable_names:
        return {}

    query = """
        SELECT variable_ID, name
        FROM raw.census_variables
        WHERE LOWER(TRIM(name)) IN (SELECT UNNEST(?))
        ORDER BY variable_ID
    """

    df = con.execute(query, [variable_names]).df()
    return dict(zip(df["variable_ID"], df["name"]))

###########################################
#Final Query Functions#####################
###########################################
###########################################
def get_census_facts_wide(
    con,
    ids: list[str],                  # Expect 5-digit county FIPS like '06037'
    variable_dict: dict[str, str],
    Type: str = "state",
) -> pd.DataFrame:
    variable_ids = list(variable_dict.keys())
    type_lower = (Type or "state").strip().lower()

    if type_lower == "county":
        # Enforce/normalize to 5-digit FIPS inside SQL; filters and outputs are 5-digit with leading zeros.
        sql = """
        WITH
          nf AS (  -- normalize incoming ids to exactly 5-digit zero-padded strings
            SELECT LPAD(REGEXP_REPLACE(CAST(x AS VARCHAR), '[^0-9]', ''), 5, '0') AS county_id_5
            FROM (SELECT UNNEST(?) AS x)
          ),
          ns AS (  -- states with normalized state_id (strip leading zeros, but output re-padded to 2)
            SELECT s.*,
                   REGEXP_REPLACE(CAST(s.state_ID AS VARCHAR), '^0+', '') AS state_id_norm
            FROM raw.states s
          ),
          nc AS (  -- counties with normalized state_id
            SELECT c.*,
                   REGEXP_REPLACE(CAST(c.state_ID AS VARCHAR), '^0+', '') AS state_id_norm
            FROM raw.counties c
          )
        SELECT
          LPAD(ns.state_id_norm, 2, '0')                     AS state_id,
          ns.name                                            AS state_name,
          LPAD(CAST(nc.County_ID AS VARCHAR), 5, '0')        AS county_id,   -- always 5-digit
          nc.name                                            AS county_name,
          v.variable_ID                                      AS variable_id,
          v.name                                             AS variable_name,
          CAST(cf.data AS DOUBLE)                            AS value
        FROM ns
        JOIN nc
          ON nc.state_id_norm = ns.state_id_norm
        JOIN raw.census_facts cf
          ON cf.County_ID = nc.County_ID
        JOIN raw.census_variables v
          ON v.variable_ID = cf.variable_ID
        WHERE LPAD(CAST(nc.County_ID AS VARCHAR), 5, '0') IN (SELECT county_id_5 FROM nf)
          AND v.variable_ID IN (SELECT UNNEST(?))
        ORDER BY ns.state_id_norm, LOWER(nc.name), v.variable_ID;
        """
        params = [ids, variable_ids]

    else:
        # State branch unchanged in logic, but always emit county_id as zero-padded 5-digit.
        sql = """
        WITH
          ns AS (
            SELECT s.*,
                   REGEXP_REPLACE(CAST(s.state_ID AS VARCHAR), '^0+', '') AS state_id_norm
            FROM raw.states s
          ),
          nc AS (
            SELECT c.*,
                   REGEXP_REPLACE(CAST(c.state_ID AS VARCHAR), '^0+', '') AS state_id_norm
            FROM raw.counties c
          ),
          nf AS (  -- normalize incoming state ids (strip leading zeros)
            SELECT REGEXP_REPLACE(CAST(x AS VARCHAR), '^0+', '') AS state_id_norm
            FROM (SELECT UNNEST(?) AS x)
          )
        SELECT
          LPAD(ns.state_id_norm, 2, '0')               AS state_id,
          ns.name                                       AS state_name,
          LPAD(CAST(nc.County_ID AS VARCHAR), 5, '0')   AS county_id,  -- always 5-digit
          nc.name                                       AS county_name,
          v.variable_ID                                 AS variable_id,
          v.name                                        AS variable_name,
          CAST(cf.data AS DOUBLE)                       AS value
        FROM ns
        JOIN nc
          ON nc.state_id_norm = ns.state_id_norm
        JOIN raw.census_facts cf
          ON cf.County_ID = nc.County_ID
        JOIN raw.census_variables v
          ON v.variable_ID = cf.variable_ID
        WHERE ns.state_id_norm IN (SELECT state_id_norm FROM nf)
          AND v.variable_ID IN (SELECT UNNEST(?))
        ORDER BY ns.state_id_norm, LOWER(nc.name), v.variable_ID;
        """
        params = [ids, variable_ids]

    df = con.execute(sql, params).df()

    # Pivot to wide; rename variable columns to "ID: name"
    if df.empty:
        base_cols = ["state_id", "state_name", "county_id", "county_name"]
        return pd.DataFrame(columns=base_cols + [f"{vid}: {variable_dict[vid]}" for vid in variable_ids])

    wide = df.pivot(
        index=["state_id", "state_name", "county_id", "county_name"],
        columns="variable_id",
        values="value"
    ).reset_index()

    wide.columns.name = None
    rename_map = {vid: f"{vid}: {variable_dict[vid]}" for vid in variable_ids if vid in wide.columns}
    wide = wide.rename(columns=rename_map)
    return wide

def get_wide_vars_by_state(
    con,
    state_ids: list[str],
    variable_names,  # may be list[str], str, list[list[str]], or dict
    table_fullname: str = "raw.state_MoVE_data",
    state_col: str = "state_ID",
) -> pd.DataFrame:
    # --- normalize state_ids to a list[str]
    if isinstance(state_ids, str):
        state_ids = [state_ids]

    # --- normalize variable_names into a flat list[str]
    # supports: str, list[str], list[list[str]], set/tuple, dict (uses keys)
    if isinstance(variable_names, dict):
        variable_names = list(variable_names.keys())
    elif isinstance(variable_names, str):
        # allow comma-separated string
        variable_names = [v.strip() for v in variable_names.split(",")]
    elif isinstance(variable_names, Iterable):
        # flatten one level if items are themselves iterables (e.g., [['A','B'], 'C'])
        flat = []
        for item in variable_names:
            if isinstance(item, str):
                flat.append(item)
            elif isinstance(item, Iterable) and not isinstance(item, (bytes, bytearray)):
                flat.extend([str(x) for x in item])
            else:
                flat.append(str(item))
        variable_names = flat
    else:
        variable_names = [str(variable_names)]

    # strip empties and ensure strings
    variable_names = [str(c).strip() for c in variable_names if str(c).strip()]

    # --- safe identifier quoting (cast to str to avoid .replace on non-strings)
    def qi(name) -> str:
        s = str(name)
        return '"' + s.replace('"', '""') + '"'

    # build SELECT clause in requested order
    select_cols = [qi(state_col)] + [qi(c) for c in variable_names]
    select_clause = ", ".join(select_cols)

    sql = f"""
        SELECT {select_clause}
        FROM {table_fullname}
        WHERE {qi(state_col)} IN (SELECT UNNEST(?))
        ORDER BY {qi(state_col)};
    """

    return con.execute(sql, [state_ids]).df()

import pandas as pd

def get_category_scores_by_states(
    con,
    state_ids: list[str],          # if Type=="county", this is a list of county_ids
    categories: list[str],
    suffix: str = "_Sum",
    Type: str = "state",
) -> pd.DataFrame:
    """
    Sum the scores of each specified category for every county and return wide format.
    Modes:
      - Type == "state": 'state_ids' are state_IDs; fixes leading-zero mismatches; outputs 2-digit state_id.
      - Type == "county": 'state_ids' is a list of County_IDs; strips leading zeros from both state_id and county_id in output.
    Output columns: state_id, state_name, county_id, county_name, <category>+suffix...
    """
    if not state_ids or not categories:
        return pd.DataFrame(columns=["state_id", "state_name", "county_id", "county_name"] + [c + suffix for c in categories])

    type_lower = (Type or "state").strip().lower()

    if type_lower == "county":
        # Filter by specific county IDs; normalize leading zeros for both state_id and county_id
        sql = """
        WITH
        nf AS (  -- normalize incoming county ids like '06037' -> '6037'
            SELECT REGEXP_REPLACE(CAST(x AS VARCHAR), '^0+', '') AS county_id_norm
            FROM (SELECT UNNEST(?) AS x)
        ),
        ns AS (  -- states with normalized state_id (to join with counties)
            SELECT s.*,
                REGEXP_REPLACE(CAST(s.state_ID AS VARCHAR), '^0+', '') AS state_id_norm
            FROM raw.states s
        ),
        nc AS (  -- counties with normalized ids
            SELECT c.*,
                REGEXP_REPLACE(CAST(c.state_ID  AS VARCHAR), '^0+', '') AS state_id_norm,
                REGEXP_REPLACE(CAST(c.County_ID AS VARCHAR), '^0+', '') AS county_id_norm
            FROM raw.counties c
        )
        SELECT
        LPAD(ns.state_id_norm, 2, '0')            AS state_id,      -- ensure 2-digit state FIPS
        ns.name                                   AS state_name,
        LPAD(nc.county_id_norm, 5, '0')           AS county_id,     -- ensure 5-digit county FIPS
        nc.name                                   AS county_name,
        q.category                                AS category,
        SUM(CAST(r.value AS DOUBLE))              AS county_category_score
        FROM nf
        JOIN nc ON nc.county_id_norm = nf.county_id_norm
        JOIN ns ON ns.state_id_norm = nc.state_id_norm
        JOIN raw.responses  r ON r.County_ID = nc.County_ID
        JOIN raw.questions  q ON q.Question_ID = r.Question_ID
        WHERE q.category IN (SELECT UNNEST(?))
        GROUP BY ns.state_id_norm, ns.name, nc.county_id_norm, nc.name, q.category
        ORDER BY ns.state_id_norm, LOWER(nc.name), q.category;
"""

        params = [state_ids, categories]
    else:
        # Filter by state IDs; normalize and output canonical 2-digit state_id
        sql = """
        WITH
        nf AS (
            SELECT REGEXP_REPLACE(CAST(x AS VARCHAR), '^0+', '') AS state_id_norm
            FROM (SELECT UNNEST(?) AS x)
        ),
        ns AS (
            SELECT s.*,
                REGEXP_REPLACE(CAST(s.state_ID AS VARCHAR), '^0+', '') AS state_id_norm
            FROM raw.states s
        ),
        nc AS (
            SELECT c.*,
                REGEXP_REPLACE(CAST(c.state_ID AS VARCHAR), '^0+', '') AS state_id_norm
            FROM raw.counties c
        )
        SELECT
        LPAD(ns.state_id_norm, 2, '0')        AS state_id,        -- ensure 2-digit state FIPS
        ns.name                               AS state_name,
        LPAD(CAST(nc.County_ID AS VARCHAR), 5, '0') AS county_id, -- ensure 5-digit county FIPS
        nc.name                               AS county_name,
        q.category                            AS category,
        SUM(CAST(r.value AS DOUBLE))          AS county_category_score
        FROM nf
        JOIN ns ON ns.state_id_norm = nf.state_id_norm
        JOIN nc ON nc.state_id_norm = ns.state_id_norm
        JOIN raw.responses  r ON r.County_ID = nc.County_ID
        JOIN raw.questions  q ON q.Question_ID = r.Question_ID
        WHERE q.category IN (SELECT UNNEST(?))
        GROUP BY ns.state_id_norm, ns.name, nc.County_ID, nc.name, q.category
        ORDER BY ns.state_id_norm, LOWER(nc.name), q.category;
        """

        params = [state_ids, categories]

    df_long = con.execute(sql, params).df()
    if df_long.empty:
        return pd.DataFrame(columns=["state_id", "state_name", "county_id", "county_name"] + [c + suffix for c in categories])

    # Pivot to wide format
    wide = df_long.pivot(
        index=["state_id", "state_name", "county_id", "county_name"],
        columns="category",
        values="county_category_score"
    ).reset_index()

    # Ensure all requested categories exist and apply suffix
    wide.columns.name = None
    for c in categories:
        if c not in wide.columns:
            wide[c] = 0.0

    rename_map = {c: c + suffix for c in categories if c in wide.columns}
    wide = wide.rename(columns=rename_map)

    # Column order: meta + requested categories (with suffix)
    meta = ["state_id", "state_name", "county_id", "county_name"]
    wide = wide[meta + [c + suffix for c in categories]]

    # Ensure numeric
    for col in [c + suffix for c in categories]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(0.0)

    return wide

def get_question_matrix_by_states(
    con,
    ids: list[str],
    question_ids: list[str],
    total_col: str = "selected_total",
    Type: str = "state",
) -> pd.DataFrame:
    """
    Build a county × selected-questions matrix for either:
      - Type == "state": ids = state IDs (with or without leading zeros)
      - Type == "county": ids = county IDs (with or without leading zeros)

    Output columns:
      state_id, state_name, county_id, county_name, <question_name...>, <total_col>
    """
    if not ids or not question_ids:
        return pd.DataFrame(columns=["state_id", "state_name", "county_id", "county_name", total_col])

    mode = (Type or "state").strip().lower()

    if mode == "county":
        # Filter by county IDs -> normalize both county & state IDs for joins;
        # output padded IDs (2-digit state, 5-digit county)
        sql = """
        WITH
          nf AS (  -- normalize incoming county ids like '06037' -> '6037'
            SELECT REGEXP_REPLACE(CAST(x AS VARCHAR), '^0+', '') AS county_id_norm
            FROM (SELECT UNNEST(?) AS x)
          ),
          ns AS (  -- states with normalized state_id (to join with counties)
            SELECT s.*,
                   REGEXP_REPLACE(CAST(s.state_ID AS VARCHAR), '^0+', '') AS state_id_norm
            FROM raw.states s
          ),
          nc AS (  -- counties with normalized ids
            SELECT c.*,
                   REGEXP_REPLACE(CAST(c.state_ID  AS VARCHAR), '^0+', '') AS state_id_norm,
                   REGEXP_REPLACE(CAST(c.County_ID AS VARCHAR), '^0+', '') AS county_id_norm
            FROM raw.counties c
          ),
          sel_counties AS (
            SELECT
              LPAD(ns.state_id_norm, 2, '0')      AS state_id,     -- 2-digit
              ns.name                              AS state_name,
              LPAD(nc.county_id_norm, 5, '0')      AS county_id,    -- 5-digit
              nc.name                              AS county_name,
              nc.County_ID                         AS county_id_raw
            FROM nc
            JOIN ns ON ns.state_id_norm = nc.state_id_norm
            JOIN nf ON nf.county_id_norm = nc.county_id_norm
          ),
          sel_questions AS (
            SELECT q.Question_ID AS question_id, q.question AS question_name
            FROM raw.questions q
            WHERE q.Question_ID IN (SELECT UNNEST(?))
          ),
          long AS (
            SELECT
              sc.state_id,
              sc.state_name,
              sc.county_id,
              sc.county_name,
              sq.question_id,
              sq.question_name,
              COALESCE(SUM(CAST(r.value AS DOUBLE)), 0.0) AS value_sum
            FROM sel_counties sc
            CROSS JOIN sel_questions sq
            LEFT JOIN raw.responses r
              ON r.County_ID = sc.county_id_raw
             AND r.Question_ID = sq.question_id
            GROUP BY sc.state_id, sc.state_name, sc.county_id, sc.county_name, sq.question_id, sq.question_name
          )
        SELECT *
        FROM long
        ORDER BY state_id, LOWER(county_name), question_id;
        """
        params = [ids, question_ids]

    else:
        # Default: filter by state IDs (normalize leading zeros for joins); output 2-digit state ID
        sql = """
        WITH
          nf AS (  -- normalize incoming state ids like '06' -> '6'
            SELECT REGEXP_REPLACE(CAST(x AS VARCHAR), '^0+', '') AS state_id_norm
            FROM (SELECT UNNEST(?) AS x)
          ),
          ns AS (
            SELECT s.*,
                   REGEXP_REPLACE(CAST(s.state_ID AS VARCHAR), '^0+', '') AS state_id_norm
            FROM raw.states s
          ),
          nc AS (
            SELECT c.*,
                   REGEXP_REPLACE(CAST(c.state_ID AS VARCHAR), '^0+', '') AS state_id_norm
            FROM raw.counties c
          ),
          sel_counties AS (
            SELECT
              LPAD(ns.state_id_norm, 2, '0') AS state_id,  -- 2-digit
              ns.name AS state_name,
              LPAD(REGEXP_REPLACE(CAST(nc.County_ID AS VARCHAR), '^0+', ''), 5, '0') AS county_id, -- 5-digit
              nc.name AS county_name,
              nc.County_ID AS county_id_raw
            FROM ns
            JOIN nf ON nf.state_id_norm = ns.state_id_norm
            JOIN nc ON nc.state_id_norm = ns.state_id_norm
          ),
          sel_questions AS (
            SELECT q.Question_ID AS question_id, q.question AS question_name
            FROM raw.questions q
            WHERE q.Question_ID IN (SELECT UNNEST(?))
          ),
          long AS (
            SELECT
              sc.state_id,
              sc.state_name,
              sc.county_id,
              sc.county_name,
              sq.question_id,
              sq.question_name,
              COALESCE(SUM(CAST(r.value AS DOUBLE)), 0.0) AS value_sum
            FROM sel_counties sc
            CROSS JOIN sel_questions sq
            LEFT JOIN raw.responses r
              ON r.County_ID = sc.county_id_raw
             AND r.Question_ID = sq.question_id
            GROUP BY sc.state_id, sc.state_name, sc.county_id, sc.county_name, sq.question_id, sq.question_name
          )
        SELECT *
        FROM long
        ORDER BY state_id, LOWER(county_name), question_id;
        """
        params = [ids, question_ids]

    # Execute and pivot
    df_long = con.execute(sql, params).df()
    if df_long.empty:
        return pd.DataFrame(columns=["state_id", "state_name", "county_id", "county_name", total_col])

    wide = df_long.pivot(
        index=["state_id", "state_name", "county_id", "county_name"],
        columns="question_name",
        values="value_sum",
    ).reset_index()
    wide.columns.name = None

    # Build id->name map to preserve the passed order
    id_to_name = (
        df_long[["question_id", "question_name"]]
        .drop_duplicates()
        .set_index("question_id")["question_name"]
        .to_dict()
    )
    qcols_ordered = [id_to_name[qid] for qid in question_ids if qid in id_to_name]

    # Ensure all requested question columns exist
    for qname in qcols_ordered:
        if qname not in wide.columns:
            wide[qname] = 0.0

    # Final ordering: meta + question cols + total
    meta = ["state_id", "state_name", "county_id", "county_name"]
    wide = wide[meta + qcols_ordered]

    # Total column across the chosen questions
    wide[total_col] = wide[qcols_ordered].sum(axis=1, skipna=True)

    # Ensure numeric
    for c in qcols_ordered + [total_col]:
        wide[c] = pd.to_numeric(wide[c], errors="coerce").fillna(0.0)

    return wide

def get_county_move_vars(
    con,
    ids: list[str],
    county_move_vars: dict,
    Type: str = "state"
) -> pd.DataFrame:
    # build list of all variable names to select
    sum_score_vars = []
    df_final = pd.DataFrame()
    for category, details in county_move_vars.items():
        if details.get("sum_scores", True):
            sum_score_vars.append(category)
        else:
            selected_questions = details.get("selected_questions", [])
            df = get_question_matrix_by_states(con, ids, get_question_ids(con, selected_questions), total_col=f"{category}_Sum", Type=Type)
            if df_final.empty:
                df_final = df
            else:
                df_final = df_final.merge(
                    df,
                    on=["state_id", "county_id"],
                    how="outer",            # or "left" if you only want existing rows
                    suffixes=("", "_dup"),
                )
    if sum_score_vars:
        sum_scores = get_category_scores_by_states(con, ids, sum_score_vars, Type=Type)
        if df_final.empty:
            df_final = sum_scores
        else:
            df_final = df_final.merge(
                sum_scores,
                on=["state_id", "county_id"],
                how="outer",
                suffixes=("", "_dup"))
    dup_cols = [c for c in df_final.columns if c.endswith("_dup")]
    df_final = df_final.drop(columns=dup_cols)
    return df_final

def run_all_state_query(
    con,
    state_ids: list[str],
    census_vars: dict | None,
    state_move_vars: list[str] | None,
    county_move_vars: dict | None,
) -> pd.DataFrame:
    # --- Run queries ---
    census_df = get_census_facts_wide(con, state_ids, census_vars) if census_vars else pd.DataFrame()
    county_move_df = get_county_move_vars(con, state_ids, county_move_vars) if county_move_vars else pd.DataFrame()
    state_move_df = get_wide_vars_by_state(con, state_ids, state_move_vars) if state_move_vars else pd.DataFrame()
    # --- Merge county-level frames on ["state_id","county_id"] ---
    county_frames = [df for df in (census_df, county_move_df) if not df.empty]
    if county_frames:
        df_final = reduce(
            lambda L, R: pd.merge(L, R, on=["state_id", "county_id"], how="outer", suffixes=("", "_dup")),
            county_frames
        )
        # drop any dup columns from the right-hand side merges
        dup_cols = [c for c in df_final.columns if c.endswith("_dup")]
        if dup_cols:
            df_final = df_final.drop(columns=dup_cols)
        # if any exact duplicates slipped through, keep the first
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    else:
        # nothing county-level; start from empty
        df_final = pd.DataFrame()
    # --- Finally, if state_move_df exists, merge it by state_id only ---
    if not state_move_df.empty:
        if df_final.empty:
            # nothing to broadcast onto—just return state-level frame
            return state_move_df
        df_final = df_final.merge(state_move_df, on="state_id", how="left", suffixes=("", "_dup"))
        dup_cols = [c for c in df_final.columns if c.endswith("_dup")]
        if dup_cols:
            df_final = df_final.drop(columns=dup_cols)
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    return df_final


def run_county_query(
    con,
    county_FIPs: list[str],
    census_vars: dict | None,
    state_move_vars: list[str] | None,
    county_move_vars: dict | None,
) -> pd.DataFrame:
    # --- Run queries ---
    census_df = get_census_facts_wide(con, county_FIPs, census_vars, Type = "county") if census_vars else pd.DataFrame()
    county_move_df = get_county_move_vars(con, county_FIPs, county_move_vars, Type="county") if county_move_vars else pd.DataFrame()
    if state_move_vars:
        state_ids = state_ids_from_counties(con, county_FIPs)
        state_move_df = get_wide_vars_by_state(con, state_ids, state_move_vars)
    else:
        state_move_df = pd.DataFrame()
    county_frames = [df for df in (census_df, county_move_df) if not df.empty]
    if county_frames:
        df_final = reduce(
            lambda L, R: pd.merge(L, R, on=["state_id", "county_id"], how="outer", suffixes=("", "_dup")),
            county_frames
        )
        # drop any dup columns from the right-hand side merges
        dup_cols = [c for c in df_final.columns if c.endswith("_dup")]
        if dup_cols:
            df_final = df_final.drop(columns=dup_cols)
        # if any exact duplicates slipped through, keep the first
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    else:
        # nothing county-level; start from empty
        df_final = pd.DataFrame()
    # --- Finally, if state_move_df exists, merge it by state_id only ---
    if not state_move_df.empty:
        if df_final.empty:
            # nothing to broadcast onto—just return state-level frame
            return state_move_df
        df_final = df_final.merge(state_move_df, on="state_id", how="left", suffixes=("", "_dup"))
        dup_cols = [c for c in df_final.columns if c.endswith("_dup")]
        if dup_cols:
            df_final = df_final.drop(columns=dup_cols)
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    return df_final