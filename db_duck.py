# db_duck.py
from contextlib import contextmanager
import duckdb
from pathlib import Path
import os, tempfile, requests, duckdb, streamlit as st

@st.cache_data(ttl=600, show_spinner=True)
def get_snapshot(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".duckdb")
    os.write(fd, r.content)
    os.close(fd)
    return path

@st.cache_resource
def connect_ro(db_path: str):
    return duckdb.connect(db_path, read_only=True)

def establish_db_connection():
    try:
        # Attempt to load the URL from secrets
        snapshot_url = st.secrets.get("SNAPSHOT_URL", "")
        if not snapshot_url:
            st.error("❌ No snapshot URL found. Please check .streamlit/secrets.toml.")
            return None

        # Try to download the snapshot file
        db_path = get_snapshot(snapshot_url)

        # Try to connect to the downloaded snapshot
        con = connect_ro(db_path)
        st.success("✅ Connected to database successfully.")
        return con

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error while downloading database: {e}")
        return None

    except duckdb.IOException as e:
        st.error(f"❌ Database connection error: {e}")
        return None

    except Exception as e:
        st.error(f"❌ Unexpected error establishing connection: {e}")
        return None
