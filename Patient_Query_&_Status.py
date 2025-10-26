# Filename: 0_❓_Patient_Query_&_Status.py

import streamlit as st
from datetime import datetime
# Import necessary functions from utils
from utils import (
    init_json_dbs,
    init_surgery_db,
    get_patient_dashboard,
    draw_chat_history,
    draw_sidebar,
    handle_audio_processing,
    run_query_agent, # Use the query-specific agent
    check_autoplay   # Import autoplay handler
)

# --- Page Config & Title ---
st.set_page_config(
    # This sets the browser tab title
    page_title="Patient Query & Status",
    layout="wide",
    page_icon="🏥" # Sets the icon in the tab and sidebar
)
check_autoplay() # Handles audio playback after rerun

# --- Initialize Databases ---
# These functions run once and are cached
init_json_dbs()
init_surgery_db()

# This sets the title shown on the page itself
st.title("Patient Query & Status ❓")
st.caption("Ask questions about patients. I will retrieve data from admissions, rooms, and surgery notes.")

# --- Draw Patient Dashboard ---
with st.expander("Show/Hide Current Patient List"):
    patient_df = get_patient_dashboard()
    if not patient_df.empty:
        # Displaying with hide_index=True for cleaner look
        st.dataframe(patient_df, use_container_width=True, hide_index=True)
    else:
        st.write("No patient data available.")

# --- Draw UI Components ---
# Pass a unique key for this page's state
page_key = "query_page"
audio_bytes, selected_lang_code = draw_sidebar(page_key=page_key)
draw_chat_history(page_key=page_key)

# --- Run Core Logic ---
if audio_bytes:
    # Pass the 'run_query_agent' and page key to the handler
    handle_audio_processing(run_query_agent, audio_bytes, selected_lang_code, page_key=page_key)