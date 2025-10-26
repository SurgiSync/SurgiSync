import streamlit as st
import pandas as pd # Needed for styling
from datetime import datetime
# Import necessary functions from utils
from utils import (
    init_json_dbs,
    init_surgery_db,
    get_room_status_dashboard,
    draw_chat_history,
    draw_sidebar,
    handle_audio_processing,
    run_management_agent, # Use the management-specific agent
    check_autoplay        # Import autoplay handler
)

# --- Page Config & Title ---
st.set_page_config(
    page_title="IPD/OPD - (Nurse/Admin)",
    layout="wide",
    page_icon="🏥"
)
check_autoplay() # <-- Call autoplay handler

# --- Initialize Databases & Session State ---
init_json_dbs()
init_surgery_db()

st.title("🏥 IPD/OPD - (Nurse/Admin)")
st.caption("Use this page to admit, discharge, transfer, or get the status of patients.")

# --- 2. Draw Dashboard ---
with st.expander("Show/Hide Live Room Status"):
    # 1. Get the raw data
    room_df = get_room_status_dashboard()
    # 2. Define the styling function
    def highlight_occupied(val):
        # Check explicitly for 'yes', handle None or other values as green
        color = 'red' if val == 'yes' else 'green'
        return f'color: {color}'
    # 3. Apply the style and display
    if not room_df.empty:
        # Apply styling using map on the specific column
        st.dataframe(
            room_df.style.map(highlight_occupied, subset=['occupied']),
            use_container_width=True,
            hide_index=True # Hide the default index column
        )
    else:
        st.write("No room data to display.")


# --- 3. Draw UI Components ---
page_key = "management_page" # Unique key for this page
audio_bytes, selected_lang_code = draw_sidebar(page_key=page_key)
draw_chat_history(page_key=page_key) # Draws the chat history specific to this page

# --- 4. Run Core Logic ---
if audio_bytes:
    # Pass the 'run_management_agent' function to the handler
    handle_audio_processing(run_management_agent, audio_bytes, selected_lang_code, page_key=page_key)