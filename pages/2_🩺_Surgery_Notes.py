import streamlit as st
from datetime import datetime
# Import necessary functions from utils
from utils import (
    init_json_dbs,
    init_surgery_db,
    get_recent_surgery_notes,
    draw_chat_history,
    draw_sidebar,
    handle_audio_processing,
    run_surgery_agent,     # Use the surgery-specific agent
    check_autoplay       # Import autoplay handler
)

# --- Page Config & Title ---
st.set_page_config(
    page_title="Surgery/Ward Notes and Transcript",
    layout="wide",
    page_icon="🩺"
)
check_autoplay() # <-- Call autoplay handler

# --- Initialize Databases & Session State ---
init_json_dbs()
init_surgery_db()

st.title("🩺 Surgery/Ward Notes and Transcripts")
st.caption("Use this page to record surgery comments. Start by saying 'Record comment for patient 101...'.")

# --- 2. Draw Dashboard ---
with st.expander("Show/Hide Recent Surgery Notes"):
    notes_df = get_recent_surgery_notes()
    if not notes_df.empty:
        # Displaying with hide_index=True for cleaner look
        st.dataframe(notes_df, use_container_width=True, hide_index=True)
    else:
        st.write("No surgery notes recorded yet.")


# --- 3. Draw UI Components ---
page_key = "surgery_page" # Unique key for this page
audio_bytes, selected_lang_code = draw_sidebar(page_key=page_key)
draw_chat_history(page_key=page_key)

# --- 4. Run Core Logic ---
if audio_bytes:
    # Pass the 'run_surgery_agent' function to the handler
    handle_audio_processing(run_surgery_agent, audio_bytes, selected_lang_code, page_key=page_key)