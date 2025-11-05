import streamlit as st
import pandas as pd
import re
from pathlib import Path
import sys, os
###########################################
#MoVE and Census variable checkboxes logic#
###########################################
def checkbox_multi_census(label: str, options: list[str], cols: int = 3) -> list[str]:
    st.subheader(label)

    q = st.text_input("Search census variables", placeholder="Type to filter…").strip().lower()
    filtered = [o for o in options if q in o.lower()] if q else options

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Select all", key=f"select_all_{label}"):
            for i, opt in enumerate(filtered):
                st.session_state[f"var_census_{opt}"] = True
    with c2:
        if st.button("Clear all", key=f"clear_all_{label}"):
            for i, opt in enumerate(filtered):
                st.session_state[f"var_census_{opt}"] = False

    cols_list = st.columns(cols)
    for i, opt in enumerate(filtered):
        with cols_list[i % cols]:
            st.checkbox(
                opt,
                key=f"var_census_{opt}",
                value=st.session_state.get(f"var_census_{opt}", opt == "Overall Population")
            )

    selected = [opt for opt in options if st.session_state.get(f"var_census_{opt}", False)]
    return selected

def checkbox_multi_MoVE(label: str, dataframe, cols: int = 3):
    category_map = {
        "Reg": "The Provision of Information About Registration (5 item additive scale)", 
        "Voting": "The Provision of Information About Voting", 
        "Abuses": "History of Voting Rights Abuses in County (3 item Likert scale)",
        "pollworkers": "Availability of Poll Workers (6 item additive scale)",
        "Registration": "Registration Drives (dichotomous 0/1)",
        "dropboxes": "Alternative Voting: Drop Boxes (dichotomous 0/1)",
        "Vote Centers": "Alternative Voting: Vote Centers (dichotomous 0/1)",
        "Ease of Registration": "Ease of Registration (4 item additive scale)"
    }

    # list of category keys present in the dataframe
    filtered = list(dataframe["category"].unique())
    st.subheader(label)

    # --- Select all / clear all buttons ---
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Select all", key=f"select_all_{label}"):
            for opt in filtered:
                st.session_state[f"var_cb_{opt}"] = True
    with c2:
        if st.button("Clear all", key=f"clear_all_{label}"):
            for opt in filtered:
                st.session_state[f"var_cb_{opt}"] = False

    # --- Render checkbox grid ---
    cols_list = st.columns(max(1, min(cols, len(filtered))))
    for i, opt in enumerate(filtered):
        display_label = category_map.get(opt, opt)
        with cols_list[i % len(cols_list)]:
            checked = st.checkbox(
                display_label,
                key=f"var_cb_{opt}",
                value=st.session_state.get(f"var_cb_{opt}", False),
                help=f"Key: {opt}"
            )
            if checked:
                st.caption("Clicking Sum Scores will count all scores in this category and only return the total score in your export.")
                sum_key = f"adv_{opt}"
                detail_key = f"detail_{opt}"

                use_advanced = st.toggle(f"Sum Scores ({opt})", value=True, key=sum_key)

                sub_opts = dataframe.loc[dataframe["category"] == opt, "question"].tolist()

                if use_advanced:
                    # clear details when sum mode is ON
                    if detail_key in st.session_state:
                        del st.session_state[detail_key]
                else:
                    # render multiselect with prior selection if any
                    prev = st.session_state.get(detail_key, [])
                    st.multiselect(
                        "Grab specific scores",
                        options=sub_opts,
                        default=prev,            # only used on first render; then key keeps state
                        key=detail_key
                    )


    # --- Return list of selected keys (short names) ---
    selected = [opt for opt in filtered if st.session_state.get(f"var_cb_{opt}", False)]
    return selected

#################################
#Selection summary rendering####
#################################
def build_selection_summary_dict() -> dict:
    """Collect current selections into a simple nested dict."""
    summary = {
        "States We're Exporting": st.session_state.get("state_abrrev", "Specific Counties"),
        "Counties We're Exporting": st.session_state.get("county_names", "All in State(s)"),
        "Census variables": st.session_state.get("sel_census", []),
        "State MoVE variables": st.session_state.get("sel_state_move", []),
        "County MoVE variables": {}
    }
    county = {}
    for key, val in st.session_state.items():
        if key.startswith("var_cb_") and val is True:
            category = key[len("var_cb_"):]
            county[category] = {
                "sum_scores": bool(st.session_state.get(f"adv_{category}", True)),
                "selected_questions": st.session_state.get(f"detail_{category}", [])
            }
    summary["County MoVE variables"] = county

    return summary

def render_selection_summary():
    st.subheader("Your selections (JSON)")
    st.json(build_selection_summary_dict())
    

###################################
#Attach Headers to columns#########
###################################
def _normalize(col: str) -> str:
    """
    Normalize column names for mapping:
      - remove leading '<digits>:'
      - strip trailing '_sum'
      - trim whitespace
    """
    if col is None:
        return col
    s = str(col).strip()
    s = re.sub(r'^\d+\s*:\s*', '', s)     # drop '1:' or '2:' or '12:' etc.
    s = re.sub(r'_Sum$', '', s)           # drop trailing '_sum'
    return s.strip()

def apply_single_header(df: pd.DataFrame, csv_path: str, default="Other") -> pd.DataFrame:
    """
    Reads a CSV with columns: column, header
    Builds a 2-level MultiIndex: [header, original_col]
    Matching is done on a *normalized* version of the column name.
    """
    meta = pd.read_csv(csv_path)
    if not {"name", "header"}.issubset(meta.columns):
        raise ValueError("CSV must have columns: 'column', 'header'")

    # Build lookup on normalized column names from the CSV
    lookup = { _normalize(c): h for c, h in zip(meta["name"], meta["header"]) }

    # Build the top header row following the *original df column order*
    top = [ lookup.get(_normalize(c), default) for c in df.columns ]

    # Optional sanity reports (based on normalized names)
    df_norm = { _normalize(c) for c in df.columns }
    csv_norm = set(lookup.keys())

    out = df.copy()
    out.columns = pd.MultiIndex.from_arrays([top, df.columns])
    return out

def base_dir() -> Path:
    """Return the base directory depending on context."""
    # When running as a PyInstaller .exe, files are unpacked to sys._MEIPASS
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    # When running normally (streamlit run app.py)
    return Path(__file__).resolve().parents[1]

def asset_path(*parts: str) -> Path:
    """Build a full path to something in the assets folder."""
    return base_dir().joinpath("assets", *parts)


########################################################
#Optional Summary Statistics Section####################
#########################################################
def show_summary_stats(df: pd.DataFrame):
    st.subheader("Summary Statistics")
    df.columns = df.columns.droplevel(0)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for summary statistics.")
        return

    selected_cols = st.multiselect(
        "Select numeric columns to summarize",
        options=numeric_cols,
        default=numeric_cols[:3]  # default to first 3 numeric columns
    )

    if not selected_cols:
        st.info("Please select at least one numeric column.")
        return

    summary_df = df[selected_cols].describe().T
    summary_df['median'] = df[selected_cols].median()
    summary_df = summary_df[['count', 'mean', 'std', 'min', '25%', '50%', 'median', '75%', 'max']]

    st.dataframe(summary_df, use_container_width=True)
