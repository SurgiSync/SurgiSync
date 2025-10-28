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
    run_query_agent      # Use the query-specific agent
)

# --- Page Config & Title ---
st.set_page_config(
    page_title="Patient Query",
    layout="wide",
    page_icon="🏥"
)

# --- FIX: WIDER SIDEBAR + CHAT BACKGROUNDS ---
st.markdown("""
<style>
    /* WIDER SIDEBAR */
    [data-testid="stSidebar"] {
        width: 400px !important;
    }
    
    /* CHAT MESSAGE STYLING */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* User (Doctor) message background */
    [data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) {
        background-color: #E0F7FA; /* Light Cyan */
    }
    
    /* Assistant (Bot) message background */
    [data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-assistant"]) {
        background-color: #F1F8E9; /* Light Green */
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Databases & Session State ---
init_json_dbs()
init_surgery_db()

st.title("❓ General Patient Query")
st.caption("Use this page to ask questions about patients (e.g., 'What's the status of patient 101?') or the hospital (e.g., 'How many beds are free?').")

# --- 2. Draw Dashboard ---
with st.expander("Show/Hide Admitted Patients List"):
    patients_df = get_patient_dashboard()
    if not patients_df.empty:
        st.dataframe(patients_df, use_container_width=True, hide_index=True)
    else:
        st.write("No patients currently admitted.")


# --- 3. Draw UI Components ---
page_key = "query_page" # Unique key for this page

# --- FIX: ADD STATE TO TRACK AUDIO (FOR LOOP PREVENTION) ---
if f'last_processed_audio_{page_key}' not in st.session_state:
    st.session_state[f'last_processed_audio_{page_key}'] = None

audio_bytes, selected_lang_code = draw_sidebar(page_key=page_key)
draw_chat_history(page_key=page_key) # Draw chat *before* processing

# --- 4. Run Core Logic (WITH NEW CHECK) ---
if audio_bytes and (st.session_state[f'last_processed_audio_{page_key}'] != audio_bytes):
    
    # 1. Store the new audio bytes to prevent reprocessing
    st.session_state[f'last_processed_audio_{page_key}'] = audio_bytes
    
    # 2. Process the audio
    handle_audio_processing(run_query_agent, audio_bytes, selected_lang_code, page_key=page_key)
    
    # 3. Rerun the page to display the new chat messages
    st.rerun()