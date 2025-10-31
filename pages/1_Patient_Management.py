import streamlit as st
import pandas as pd
from datetime import datetime
from utils import (
    init_json_dbs,
    init_surgery_db,
    get_room_status_dashboard,
    draw_chat_history,
    draw_sidebar,
    handle_audio_processing,
    run_management_agent      # Import the specific agent for this page
)

st.set_page_config(page_title="🏥 IPD/OPD - (Nurse/Admin)", layout="wide", page_icon="🏥")

# --- WIDER SIDEBAR ---
st.markdown("""<style>[data-testid="stSidebar"] {width: 400px !important;}</style>""", unsafe_allow_html=True)

# --- 1. Initialize Databases & Session State ---
init_json_dbs()
init_surgery_db()

st.title("🏥 IPD/OPD - (Nurse/Admin)")
st.caption("Use this page to admit, discharge, transfer, or get the status of patients.")

# --- 2. Draw Dashboard ---
with st.expander("Show/Hide Live Room Status"):
    room_df = get_room_status_dashboard()
    
    def highlight_occupied(val):
        color = 'red' if val == 'yes' else 'green'
        return f'color: {color}'
    
    if not room_df.empty:
        if 'occupied' in room_df.columns:
            st.dataframe(
                room_df.style.apply(lambda col: col.map(highlight_occupied) if col.name == 'occupied' else [''] * len(col), axis=0),
                use_container_width=True,
                hide_index=True 
            )
        else:
             st.dataframe(room_df, use_container_width=True, hide_index=True)
    else:
        st.write("No room data to display.")


# --- 3. Draw UI Components ---
page_key = "management_page" # Unique key for this page

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
    handle_audio_processing(run_management_agent, audio_bytes, selected_lang_code, page_key=page_key)
    
    # 3. Rerun the page to display the new chat messages
    st.rerun()