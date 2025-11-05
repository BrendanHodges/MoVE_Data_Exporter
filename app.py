import os, tempfile, requests, duckdb, streamlit as st
from pathlib import Path
from db_duck import establish_db_connection
from utils.db_utils import get_all_variable, get_columns_in_table, grab_county_names, grab_state_ids, grab_census_variable_ids, run_all_state_query, run_county_query
from utils.app_utils import checkbox_multi_census, checkbox_multi_MoVE, render_selection_summary, apply_single_header, asset_path, show_summary_stats, filter_df_by_density
import pandas as pd
################################
#Helper Functions###############
################################

def list_them(input):
    if "," in input:
        input = [s.strip() for s in input.split(",") if s.strip()]
    else:
        input = [input.strip()] if input.strip() else []
    return input

st.set_page_config(page_title="📤 Measurement of Voting Equity data Exporter", layout="wide")

##########################################
## Streamlit App Configuration and Layout
##########################################


st.markdown("## 🏛️ MoVE — Admin Data Exporter")
st.caption("Measurement of Voting Equity • Export researcher-ready snapshots.")

# --- Database connection ---
con = establish_db_connection()
if con is None:
    st.error("Database connection failed. Check credentials and try again.")
    st.stop()

# --- Preload variable names ---
census_variables_names = get_all_variable(con, "name", "raw.census_variables")
state_MoVE_variables_names = get_columns_in_table(con, "raw.state_MoVE_data")
county_MoVE_variables = get_all_variable(con, ["question", "category"], "raw.questions")

left, right = st.columns([2.2, 1])

with left:
    st.subheader("Filters and Variable Selection")

    # --- Export scope ---
    entity = st.radio(
        "Export scope",
        ("Export by State", "Export by County"),
        horizontal=True,
        key="entity_choice",
        help="Choose to export by state abbreviations or by county FIPS codes."
    )

    # --- Inputs based on scope ---
    if entity == "Export by State":
        c1, c2 = st.columns([2, 1])
        with c1:
            states = st.text_input(
                "State abbreviations",
                placeholder="CA, TX, NY",
                key="state_abrrev"
            )
        with c2:
            st.caption("Separate with commas (e.g., CA, TX, NY)")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            county_FIPs_input = st.text_input(
                "County FIPS codes",
                placeholder="06037, 48201, 36061",
                key="county_fips_input"
            )
        with c2:
            st.caption("FIPS reference: https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt")

    st.divider()

    # --- Variables grouped in tabs ---
    tab1, tab2, tab3 = st.tabs(["Census Variables", "State MoVE Variables", "County MoVE Variables"])

    with tab1:
        with st.expander("Select census variables", expanded=False):
            selected_census_variables = checkbox_multi_census("Census variables", census_variables_names, cols=3)
        st.session_state["sel_census"] = selected_census_variables
        st.caption(f"Selected: {len(selected_census_variables)}")

    with tab2:
        selected_state_MoVE_variables = st.multiselect(
            "MoVE state variables",
            state_MoVE_variables_names,
            placeholder="Start typing to search…"
        )
        st.session_state["sel_state_move"] = selected_state_MoVE_variables
        st.caption(f"Selected: {len(selected_state_MoVE_variables)}")

    with tab3:
        with st.expander("Select county MoVE variables", expanded=False):
            selected_MoVE_variables = checkbox_multi_MoVE("County MoVE variables", county_MoVE_variables, cols=1)
        st.session_state["sel_county_move"] = selected_MoVE_variables
        st.caption(f"Selected categories: {len(selected_MoVE_variables)}")

    st.divider()

    density_filter = st.radio(
        "Export scope",
        ("All", "Rural", "Urban", "Suburban"),
        horizontal=True,
        key="Density Filter",
        help="Filter out Counties by population density (people per square mile)."
    )

    # --- Run query button (safe outside of any form) ---
    run_clicked = st.button("Run query", type="primary")

    st.divider()

with right:
    st.subheader("Tips")
    st.write(
        "- Use **state abbreviations** like `CA, TX, NY`.\n"
        "- County exports accept **FIPS codes**.\n"
        "- Use the tabs to pick which variable groups to include.\n"
        "- The table below applies **MultiIndex headers** for grouped columns."
    )
    render_selection_summary()

# --- Execute query after button click ---
if run_clicked:
    all_state = True
    state_move_vars = []

    # --- Determine export type ---
    if entity == "Export by State" and st.session_state.get("state_abrrev"):
        state_ids = grab_state_ids(st.session_state["state_abrrev"], con)
    elif entity == "Export by County" and st.session_state.get("county_fips_input"):
        county_FIPs = list_them(st.session_state["county_fips_input"])
        st.session_state['county_names'] = grab_county_names(con, county_FIPs)
        state_ids = None
        all_state = False
    else:
        st.warning("Please provide at least one state abbreviation or county FIPS code.")
        st.stop()

    # --- Grab variable IDs ---
    census_vars = grab_census_variable_ids(st.session_state.get("sel_census", []), con) \
                  if st.session_state.get("sel_census") else []

    if st.session_state.get("sel_state_move"):
        state_move_vars = [st.session_state["sel_state_move"]]

    # --- Gather county selections from checkboxes ---
    county = {}
    for key, value in st.session_state.items():
        if key.startswith("var_cb_") and value is True:
            category = key[len("var_cb_"):]
            county[category] = {
                "sum_scores": bool(st.session_state.get(f"adv_{category}", True)),
                "selected_questions": st.session_state.get(f"detail_{category}", [])
            }

    # --- Run query ---
    with st.spinner("Running query…"):
        if all_state:
            df1 = run_all_state_query(con, state_ids, census_vars, state_move_vars, county)
        else:
            df1 = run_county_query(con, county_FIPs, census_vars, state_move_vars, county)

    df1 = filter_df_by_density(df1, density_filter)

    if "Registration" or "Registration_Sums" in df1.columns:
        df1.rename(columns={"Registration_Sum": "Drives_Sum"}, inplace=True)
        print(df1.columns)
        print("WHAT THE HECK")
    # --- Apply MultiIndex headers ---
    try:
        df1 = apply_single_header(df1, asset_path("db_contents.csv"))
    except Exception as e:
        st.warning(f"Header mapping skipped: {e}")

    
    # >>> Persist results for future reruns/page switches <<<
    st.session_state.df1 = df1
# --- Show results if we either just ran the query OR have prior results in session ---
df_to_show = st.session_state.get("df1", None)

st.subheader("Results")
if df_to_show is None or len(df_to_show) == 0:
    st.info("No rows returned. Adjust filters and try again.")
else:
    if ('Census Variables', 'Land Area in Square Miles') in df_to_show.columns:
        df_to_show.drop(columns=[('Census Variables', 'Land Area in Square Miles')], inplace=True)
    if ('Other', 'Population Density') in df_to_show.columns:
        df_to_show.drop(columns=[('Other', 'Population Density')], inplace=True)
    st.dataframe(df_to_show, use_container_width=True)

    # --- Optional download ---
    csv_bytes = df_to_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="move_export.csv",
        mime="text/csv",
        key="download_csv_btn"
    )

    # --- Jump to stats page; pass the DF via session_state ---
    with st.expander("Summary Statistics", expanded=False):
        st.session_state.df_for_stats = df_to_show
        df_stats = st.session_state.get("df_for_stats", None) 

        if isinstance(df_stats.columns, pd.MultiIndex) and df_stats.columns.nlevels > 1:
            df_stats.columns = df_stats.columns.droplevel(0)

        numeric_cols = df_stats.select_dtypes(include="number").columns.tolist()
        
        st.write("- Total Counties in Export:", len(df_stats))

        summary_df = df_stats[numeric_cols].describe().T
        summary_df['median'] = df_stats[numeric_cols].median()
        summary_df = summary_df[['count', 'mean', 'std', 'min', 'median', 'max']]

        st.dataframe(summary_df, use_container_width=True)
            

        





                
                
                


        

