import streamlit as st
import pandas as pd
import os

def main():
    st.set_page_config(
        page_title="Internal Database Preview",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    st.title("📊 Internal Compensation Database")
    st.markdown("""
    Here you can view the actual data used by the AI agents for compensation recommendations. This is a preview of the internal database powering the agentic workflow.
    """)
    db_path = os.path.join("data", "Compensation Data.csv")
    if os.path.exists(db_path):
        df = pd.read_csv(db_path)
        st.dataframe(df, use_container_width=True)
        st.info(f"Showing {len(df)} rows from the internal compensation database.")
    else:
        st.error("Internal database not found.")

if __name__ == "__main__":
    main()
