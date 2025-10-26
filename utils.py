import streamlit as st
import sqlite3
import pandas as pd
import base64
import os
import re
import numpy as np
import tempfile
import json
from io import BytesIO
from openai import OpenAI  # For Boson AI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder # <-- Import for sidebar

# ---
# 1. LOAD SECRETS & INITIALIZE API CLIENTS
# ---
load_dotenv()

BOSON_API_KEY = st.secrets.get("BOSON_API_KEY", os.getenv("BOSON_API_KEY"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# Ensure keys are loaded before initializing clients
if not BOSON_API_KEY or not GROQ_API_KEY:
    st.error("API keys (BOSON_API_KEY, GROQ_API_KEY) not found. Please set them in .streamlit/secrets.toml")
    # Provide dummy clients if partial functionality is desired
    boson_client = None
    groq_llm = None
else:
    boson_client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")
    groq_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

STT_MODEL_NAME = "higgs-audio-understanding-Hackathon"
TTS_MODEL_NAME = "higgs-audio-generation-Hackathon"

ROOMS_FILE = "hospital_rooms.json"
PATIENTS_FILE = "hospital_patients.json"
SURGERY_DB_FILE = "surgery_comments.db"

# ---
# 2. DATABASE INITIALIZATION & DASHBOARDS
# ---

@st.cache_resource
def init_json_dbs():
    """Initialize JSON files if they don't exist."""
    try:
        if not Path(ROOMS_FILE).exists():
            rooms_data = [
                {"id": i, "room_name": r, "bed_number": b, "occupied": "no", "pid": None}
                for i, (r, b) in enumerate([(room, bed) for room in ["A", "B", "C", "D", "E", "F"] for bed in [1, 2]], 1)
            ]
            with open(ROOMS_FILE, 'w') as f: json.dump(rooms_data, f, indent=2)
            print(f"Initialized {ROOMS_FILE}")
        if not Path(PATIENTS_FILE).exists():
            with open(PATIENTS_FILE, 'w') as f: json.dump([], f, indent=2)
            print(f"Initialized {PATIENTS_FILE}")
    except Exception as e:
        st.error(f"Error initializing JSON DBs: {e}")

@st.cache_resource
def init_surgery_db():
    """Creates the surgery_comments.db file and table if it doesn't exist."""
    conn = None
    try:
        conn = sqlite3.connect(SURGERY_DB_FILE)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS surgery_comments (
            patientid TEXT PRIMARY KEY,
            doctor TEXT,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        # Check if column exists before trying to add it
        c.execute("PRAGMA table_info(surgery_comments)")
        columns = [info[1] for info in c.fetchall()]
        if 'timestamp' not in columns:
            c.execute("ALTER TABLE surgery_comments ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
            print("Added timestamp column to surgery_comments table.")
        conn.commit()
        print(f"Initialized/Verified {SURGERY_DB_FILE}")
    except Exception as e:
        print(f"Failed to initialize surgery DB: {e}")
        st.error(f"Failed to initialize surgery DB: {e}")
    finally:
        if conn: conn.close()

@st.cache_data(ttl=30)
def get_room_status_dashboard():
    """Reads JSON and returns a raw Pandas DataFrame for the room dashboard."""
    try:
        if not Path(ROOMS_FILE).exists():
            st.warning(f"{ROOMS_FILE} not found.")
            return pd.DataFrame(columns=['id', 'room_name', 'bed_number', 'occupied', 'pid'])
        df = pd.read_json(ROOMS_FILE)
        if df.empty:
             return pd.DataFrame(columns=['id', 'room_name', 'bed_number', 'occupied', 'pid'])
        df['pid'] = df['pid'].astype('Int64')
        return df
    except Exception as e:
        st.error(f"Could not load room status: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_patient_dashboard():
    """Reads patient JSON and returns a raw Pandas DataFrame."""
    try:
        if not Path(PATIENTS_FILE).exists():
            st.warning(f"{PATIENTS_FILE} not found.")
            return pd.DataFrame(columns=['pid', 'first_name', 'last_name', 'room_name', 'bed_number', 'description', 'time_of_admit', 'dob'])
        df = pd.read_json(PATIENTS_FILE)
        if not df.empty:
             expected_cols = ['pid', 'first_name', 'last_name', 'room_name', 'bed_number', 'description', 'time_of_admit', 'dob']
             for col in expected_cols:
                 if col not in df.columns: df[col] = None
             df = df[expected_cols]
        else:
             return pd.DataFrame(columns=['pid', 'first_name', 'last_name', 'room_name', 'bed_number', 'description', 'time_of_admit', 'dob'])
        return df
    except Exception as e:
        st.error(f"Could not load patient list: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_recent_surgery_notes():
    """Reads SQL DB and returns a DataFrame of the last 5 notes."""
    try:
        if not Path(SURGERY_DB_FILE).exists():
            st.warning(f"{SURGERY_DB_FILE} not found.")
            return pd.DataFrame(columns=['timestamp', 'patientid', 'doctor', 'comment'])
        conn = sqlite3.connect(SURGERY_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='surgery_comments'")
        if cursor.fetchone() is None:
            conn.close()
            st.warning("Surgery comments table does not exist yet.")
            return pd.DataFrame(columns=['timestamp', 'patientid', 'doctor', 'comment'])

        df = pd.read_sql_query("SELECT timestamp, patientid, doctor, comment FROM surgery_comments ORDER BY timestamp DESC LIMIT 5", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Could not load recent notes: {e}")
        return pd.DataFrame()

# ---
# 3. BOSON AI API FUNCTIONS (STT & TTS)
# ---
# (Keep call_higgs_stt, call_higgs_tts, call_higgs_playback, b64_encode_bytes as they were)
def b64_encode_bytes(audio_bytes):
    return base64.b64encode(audio_bytes).decode('utf-8')

def call_higgs_stt(audio_bytes, lang_code):
    if not boson_client: return None
    st.info("Sending audio to Higgs-STT...")
    try:
        audio_b64 = b64_encode_bytes(audio_bytes)
        system = f"You are an AI assistant. Your sole task is to transcribe the user's audio into text.\nThe language of the audio is: {lang_code}."
        response = boson_client.chat.completions.create(
            model=STT_MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}]},
            ],
            modalities=["text", "audio"], max_completion_tokens=1024, timeout=60.0
        )
        st.success("Transcription received.")
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"STT Error: {e}")
        return None

def call_higgs_tts(text_with_emotion, lang_code):
    if not boson_client: return None
    st.info("Sending text to Higgs-TTS for audio generation...")
    try:
        system = f"You are an AI assistant designed to convert text into speech in language code '{lang_code}'. Respond with the appropriate emotion as tagged in the text. If no tag is present, use a calm, professional tone."
        response = boson_client.chat.completions.create(
            model=TTS_MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text_with_emotion},
            ],
            modalities=["text", "audio"],
            temperature=1.0, top_p=0.95, stream=False, timeout=300.0
        )
        audio_b64 = response.choices[0].message.audio.data
        st.success("Audio generated.")
        return base64.b64decode(audio_b64)
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def call_higgs_playback(transcript_text):
    if not boson_client: return None
    st.info("Generating multi-speaker surgery playback...")
    try:
        system = (
            "You are an audio generation AI. The user will provide a raw medical transcript. "
            "Your task is to generate a realistic, multi-speaker audio dialogue from this transcript. "
            "Assign distinct, professional-sounding voices for 'Doctor', 'Nurse', 'Anesthesiologist', etc. "
            "Read the lines with the appropriate calm, professional, or urgent tone based on the context."
        )
        response = boson_client.chat.completions.create(
            model=TTS_MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": transcript_text},
            ],
            modalities=["text", "audio"], timeout=300.0
        )
        audio_b64 = response.choices[0].message.audio.data
        st.success("Surgery playback generated.")
        return base64.b64decode(audio_b64)
    except Exception as e:
        st.error(f"TTS Playback Error: {e}")
        return None
# ---
# 4. DATABASE HELPER FUNCTIONS (READERS)
# ---
# (Keep get_context_from_dbs and find_patient_id as they were)
def get_context_from_dbs(patient_id):
    context = {}
    if patient_id is None: return json.dumps({"error": "No patient ID provided or found."})
    try:
        pid_int = int(patient_id)
        pid_str = str(patient_id)
    except ValueError: return json.dumps({"error": f"Invalid patient ID format: {patient_id}"})

    try:
        if Path(PATIENTS_FILE).exists():
            with open(PATIENTS_FILE, 'r') as f: patients_db = json.load(f)
            patient_info = next((p for p in patients_db if p.get('pid') == pid_int), None)
            context['patient_info'] = patient_info if patient_info else "No admission record found."
        else: context['patient_info'] = f"{PATIENTS_FILE} not found."
    except Exception as e: context['patient_info'] = f"Error reading patients.json: {e}"

    try:
        if Path(ROOMS_FILE).exists():
            with open(ROOMS_FILE, 'r') as f: rooms_db = json.load(f)
            room_info = next((r for r in rooms_db if r.get('pid') == pid_int), None)
            context['room_info'] = room_info if room_info else "No room assignment found."
        else: context['room_info'] = f"{ROOMS_FILE} not found."
    except Exception as e: context['room_info'] = f"Error reading rooms.json: {e}"

    try:
        if Path(SURGERY_DB_FILE).exists():
            conn = sqlite3.connect(SURGERY_DB_FILE)
            query = "SELECT doctor, comment, timestamp FROM surgery_comments WHERE patientid = ? ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn, params=(pid_str,))
            conn.close()
            context['surgery_comments'] = df.to_dict('records') if not df.empty else "No surgery comments found."
        else: context['surgery_comments'] = f"{SURGERY_DB_FILE} not found."
    except Exception as e: context['surgery_comments'] = f"Error accessing surgery database: {e}"

    return json.dumps(context, indent=2)

def find_patient_id(text):
    match = re.search(r'\b(?:patient|id)\s*(\d+)\b', text, re.IGNORECASE)
    if match: return match.group(1)
    match = re.search(r'\b(\d{3,})\b', text)
    if match: return match.group(1)
    return None

# ---
# 5. GROQ LLM "TOOL" FUNCTIONS (THE "LOGIC")
# ---
class PatientIDInfo(BaseModel):
    pid: int = Field(..., description="The patient's unique ID as an integer")

class AdmitInfo(BaseModel):
    pid: int = Field(..., description="The patient's unique ID")
    description: str = Field(..., description="A brief description of the patient's issue")

class CommentInfo(BaseModel):
    pid: int = Field(..., description="The patient's unique ID")
    doctor: str = Field(..., description="The name of the doctor, e.g., 'Dr. Aris'")
    comment: str = Field(..., description="The full surgery comment, excluding the metadata")

# --- NEW Pydantic model for Transfer ---
class TransferInfo(BaseModel):
    pid: int = Field(..., description="The patient's unique ID as an integer")
    new_room_name: str = Field(..., description="The target room letter (A-F)")
    new_bed_num: int = Field(..., description="The target bed number (1 or 2)")

    @field_validator('new_room_name')
    def validate_room(cls, v):
        if v.upper() not in ["A", "B", "C", "D", "E", "F"]:
            raise ValueError('Room name must be A, B, C, D, E, or F')
        return v.upper() # Ensure uppercase

    @field_validator('new_bed_num')
    def validate_bed(cls, v):
        if v not in [1, 2]:
            raise ValueError('Bed number must be 1 or 2')
        return v
# --- END NEW Pydantic model ---

def admit_patient_tool(note: str):
    if not groq_llm: return "[emotion: anxious] LLM client not initialized.", None
    st.info("Running: Admit Patient Tool")
    try:
        structured_llm = groq_llm.with_structured_output(AdmitInfo)
        prompt = f"Extract the patient ID (as integer) and description (as string) from this note: \"{note}\""
        info = structured_llm.invoke(prompt)
        pid = info.pid; desc = info.description

        rooms = []; patients = []
        if Path(ROOMS_FILE).exists():
             with open(ROOMS_FILE, 'r') as f: rooms = json.load(f)
        if Path(PATIENTS_FILE).exists():
             with open(PATIENTS_FILE, 'r') as f: patients = json.load(f)

        if any(p.get("pid") == pid for p in patients):
            return f"[emotion: anxious] Error: Patient {pid} is already admitted.", None

        available_room = next((r for r in rooms if r.get("occupied") == "no"), None)
        if not available_room:
            return "[emotion: sad] Error: No beds available.", None

        room_name = available_room["room_name"]; bed_num = available_room["bed_number"]
        available_room["occupied"] = "yes"; available_room["pid"] = pid

        patients.append({
            "pid": pid, "time_of_admit": datetime.now().isoformat(),
            "room_name": room_name, "bed_number": bed_num, "description": desc,
            "first_name": "Unknown", "last_name": "Unknown", "dob": "Unknown"
        })

        with open(ROOMS_FILE, 'w') as f: json.dump(rooms, f, indent=2)
        with open(PATIENTS_FILE, 'w') as f: json.dump(patients, f, indent=2)

        return f"[emotion: calm] Success. Patient {pid} admitted to room {room_name}, bed {bed_num}.", None
    except Exception as e:
        st.error(f"Admission Error details: {e}")
        return f"[emotion: anxious] Admission Error: Could not process request. Details: {e}", None

def discharge_patient_tool(text: str):
    """Discharges a patient using LLM to extract the ID."""
    if not groq_llm: return "[emotion: anxious] LLM client not initialized.", None
    st.info("Running: Discharge Patient Tool")
    try:
        structured_llm = groq_llm.with_structured_output(PatientIDInfo)
        prompt = f"Extract the patient ID as an integer from this instruction: \"{text}\""
        info = structured_llm.invoke(prompt)
        pid = info.pid

        if not pid:
             return "[emotion: anxious] Error: Could not reliably determine patient ID for discharge.", None

        rooms = []; patients = []
        if Path(ROOMS_FILE).exists():
             with open(ROOMS_FILE, 'r') as f: rooms = json.load(f)
        if Path(PATIENTS_FILE).exists():
             with open(PATIENTS_FILE, 'r') as f: patients = json.load(f)

        patient = next((p for p in patients if p.get("pid") == pid), None)
        if not patient: return f"[emotion: anxious] Error: Patient {pid} not found.", None

        room_to_free = next((r for r in rooms if r.get("pid") == pid), None)
        if room_to_free:
            room_to_free["occupied"] = "no"; room_to_free["pid"] = None

        patients = [p for p in patients if p.get("pid") != pid]

        with open(ROOMS_FILE, 'w') as f: json.dump(rooms, f, indent=2)
        with open(PATIENTS_FILE, 'w') as f: json.dump(patients, f, indent=2)

        return f"[emotion: calm] Success. Patient {pid} discharged.", None
    except Exception as e:
        st.error(f"Discharge Error details: {e}")
        return f"[emotion: anxious] Discharge Error: Could not process request. Details: {e}", None

# --- UPDATED transfer_patient_tool ---
def transfer_patient_tool(text: str):
    """Transfers a patient using LLM to extract PID, Room, and Bed."""
    if not groq_llm: return "[emotion: anxious] LLM client not initialized.", None
    st.info("Running: Transfer Patient Tool")
    try:
        # Use LLM with structured output to extract all details
        structured_llm = groq_llm.with_structured_output(TransferInfo)
        prompt = f"""
        Extract the patient ID (integer), target room letter (A-F), and target bed number (1 or 2)
        from this transfer instruction: "{text}"
        Interpret room/bed combinations like 'Room C2' or 'Bed A1' correctly.
        """
        transfer_details = structured_llm.invoke(prompt)

        pid = transfer_details.pid
        new_room_name = transfer_details.new_room_name # Already validated A-F
        new_bed_num = transfer_details.new_bed_num     # Already validated 1-2

        if not pid: # Should be caught by Pydantic, but double check
            return "[emotion: anxious] Error: Could not reliably determine patient ID for transfer.", None

        # --- Rest of the logic is the same ---
        rooms = []; patients = []
        if Path(ROOMS_FILE).exists():
             with open(ROOMS_FILE, 'r') as f: rooms = json.load(f)
        if Path(PATIENTS_FILE).exists():
             with open(PATIENTS_FILE, 'r') as f: patients = json.load(f)

        patient = next((p for p in patients if p.get("pid") == pid), None)
        if not patient:
            return f"[emotion: anxious] Error: Patient {pid} not found.", None
        old_room_name = patient["room_name"]
        old_bed_num = patient["bed_number"]

        if old_room_name == new_room_name and old_bed_num == new_bed_num:
             return f"[emotion: calm] Patient {pid} is already in Room {new_room_name} Bed {new_bed_num}.", None

        target_room = next((r for r in rooms if r.get("room_name") == new_room_name and r.get("bed_number") == new_bed_num), None)
        if not target_room:
            # This check might be redundant due to Pydantic validation, but safe
            return f"[emotion: anxious] Error: Target Room {new_room_name} Bed {new_bed_num} does not seem to exist in the system.", None

        if target_room.get("occupied") == "yes":
            return f"[emotion: anxious] Error: Target Room {new_room_name} Bed {new_bed_num} is already occupied by patient {target_room.get('pid')}.", None

        old_room = next((r for r in rooms if r.get("room_name") == old_room_name and r.get("bed_number") == old_bed_num), None)

        # Perform the transfer
        patient["room_name"] = new_room_name
        patient["bed_number"] = new_bed_num
        target_room["occupied"] = "yes"
        target_room["pid"] = pid
        if old_room:
            old_room["occupied"] = "no"
            old_room["pid"] = None

        with open(ROOMS_FILE, 'w') as f: json.dump(rooms, f, indent=2)
        with open(PATIENTS_FILE, 'w') as f: json.dump(patients, f, indent=2)

        return f"[emotion: calm] Success. Patient {pid} transferred from {old_room_name}-{old_bed_num} to {new_room_name}-{new_bed_num}.", None

    except Exception as e:
        # Catch errors from LLM extraction (like invalid room/bed) or file operations
        st.error(f"Transfer Error details: {e}")
        return f"[emotion: anxious] Transfer Error: Could not process request. {e}", None
# --- END UPDATED transfer_patient_tool ---


def record_surgery_comment_tool(text: str):
    if not groq_llm: return "[emotion: anxious] LLM client not initialized.", None
    st.info("Running: Record Surgery Comment Tool")
    try:
        structured_llm = groq_llm.with_structured_output(CommentInfo)
        prompt = f"Extract the patient ID (integer), doctor name (string), and the full comment (string) from this text. The comment is everything *after* the initial metadata (like ID and doctor name). Text: \"{text}\""
        info = structured_llm.invoke(prompt)

        pid_str = str(info.pid); doctor = info.doctor; comment = info.comment
        timestamp = datetime.now().isoformat()

        conn = sqlite3.connect(SURGERY_DB_FILE)
        c = conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO surgery_comments (patientid, doctor, comment, timestamp)
        VALUES (?, ?, ?, ?)
        """, (pid_str, doctor, comment, timestamp))
        conn.commit()
        conn.close()

        return f"[emotion: calm] Success. Surgery comment for patient {pid_str} by {doctor} has been saved.", None
    except Exception as e:
        st.error(f"Surgery Comment Error details: {e}")
        return f"[emotion: anxious] Surgery Comment Error: Could not process request. Details: {e}", None

def answer_general_query_tool(text: str):
    if not groq_llm: return "[emotion: anxious] LLM client not initialized.", None
    st.info("Running: General Query Tool")
    try:
        pid = find_patient_id(text) # Uses simple regex for query ID finding
        context = "" # Initialize context

        if not pid:
            # Check for general queries first
            general_query_keywords = ["how many", "available", "list all", "who is in", "empty beds", "occupied rooms"]
            if any(word in text.lower() for word in general_query_keywords):
                 st.info("General hospital query detected (no specific patient ID).")
                 # Load current room and patient data for context
                 rooms_df = get_room_status_dashboard()
                 patients_df = get_patient_dashboard()
                 context = f"Current Room Status:\n{rooms_df.to_string(index=False)}\n\nCurrent Patients:\n{patients_df.to_string(index=False)}"
            else:
                 # If not a general query and no PID, ask for clarification
                 return "[emotion: anxious] I'm sorry, I couldn't find a patient ID in your question. Please include the ID (e.g., 101) or ask a general question (e.g., 'How many beds are free?').", None
        else:
            st.info(f"Found Patient ID: {pid}. Fetching data from all sources...")
            context = get_context_from_dbs(pid) # Get context for the specific patient

        prompt = f"""
        You are a helpful medical assistant. Answer the user's question based *only* on the provided context.
        Be concise and accurate. If the context includes surgery comments, summarize the latest one briefly unless asked otherwise.
        If asked a general question (like 'how many beds available'), calculate the answer from the provided context table(s).

        Context:
        {context}

        Question:
        {text}

        Answer (prefix with an emotion tag [emotion: calm], [emotion: happy], or [emotion: anxious]):
        """
        response = groq_llm.invoke(prompt)
        answer = response.content.strip()

        raw_transcript = None
        # Extract transcript only if a specific patient context was loaded and valid
        if pid:
            try:
                # Safely parse context and access transcript
                context_data = json.loads(context) if isinstance(context, str) else context
                comments = context_data.get('surgery_comments')
                if isinstance(comments, list) and len(comments) > 0:
                     raw_transcript = comments[0].get('comment')
            except Exception as e:
                 st.warning(f"Could not parse context or extract transcript: {e}")


        # Ensure emotion tag exists, inferring if necessary
        if not re.match(r"\[emotion: \w+\]", answer):
            if "error" in answer.lower() or "not found" in answer.lower() or "unable" in answer.lower():
                 answer = f"[emotion: anxious] {answer}"
            else:
                 answer = f"[emotion: calm] {answer}"

        return answer, raw_transcript
    except Exception as e:
        st.error(f"Query Error details: {e}")
        return f"[emotion: anxious] Query Error: Could not process request. Details: {e}", None


# ---
# 6. PAGE-SPECIFIC AGENT ROUTERS
# ---
def run_management_agent(text_query):
    """Router for the Patient Management page."""
    if not groq_llm: return "[emotion: anxious] LLM client not initialized.", None
    st.info("Classifying management intent...")
    prompt = f"""
    You are a hospital management router. Classify the user's instruction into exactly one of:
    'admit', 'discharge', 'transfer', or 'query'.

    Instruction: "{text_query}"
    Return only the single lowercase intent word.
    """
    try:
        response = groq_llm.invoke(prompt)
        intent = response.content.strip().lower()
        st.info(f"Intent classified as: **{intent}**")

        if intent == "admit": return admit_patient_tool(text_query)
        elif intent == "discharge": return discharge_patient_tool(text_query) # Calls updated tool
        elif intent == "transfer": return transfer_patient_tool(text_query) # Calls updated tool
        elif intent == "query": return answer_general_query_tool(text_query)
        else:
            pid = find_patient_id(text_query)
            if pid:
                 st.warning("Could not classify intent clearly, defaulting to query.")
                 return answer_general_query_tool(text_query)
            else:
                 return f"[emotion: anxious] I'm sorry, I can only admit, discharge, transfer, or query patients on this page. Please state your command clearly or include a patient ID.", None
    except Exception as e:
        st.error(f"Error in management agent: {e}")
        return f"[emotion: anxious] Error classifying intent: {e}", None

def run_surgery_agent(text_query):
    """Router for the Surgery Notes page."""
    st.info("Intent forced to: **record_comment**")
    return record_surgery_comment_tool(text_query)

def run_query_agent(text_query):
    """Router for the General Query page."""
    st.info("Intent forced to: **query**")
    return answer_general_query_tool(text_query)

# ---
# 7. SHARED STREAMLIT UI COMPONENTS
# ---
# (Keep draw_chat_history, draw_sidebar, handle_audio_processing as they were,
#  but ensure handle_audio_processing does NOT have st.rerun() at the end)
def draw_chat_history(page_key=""):
    """Draws the main chat history container, using a page-specific key."""
    chat_key = f'chat_history_{page_key}'
    if chat_key not in st.session_state: st.session_state[chat_key] = []

    # Using st.container allows better control if needed, but direct drawing works
    # chat_container = st.container(height=500)
    # with chat_container:
    for message in st.session_state[chat_key]: # Iterate directly over the list
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.write(message["content"])
            # Check if transcript exists and is not None/empty before showing button
            if message.get("transcript"):
                button_key = f"play_{message['timestamp']}_{page_key}"
                if st.button(f"▶️ Playback Surgery Audio", key=button_key):
                    with st.spinner("Generating multi-speaker playback..."):
                        playback_audio = call_higgs_playback(message["transcript"])
                        if playback_audio:
                            st.audio(playback_audio, autoplay=True)


def draw_sidebar(page_key=""):
    """Draws the shared sidebar elements, using page-specific keys for state."""
    with st.sidebar:
        st.header("Controls")
        LANGUAGES = { "English": "en", "Español": "es", "Français": "fr", "Deutsch": "de", "中文": "zh" }
        selected_lang_name = st.selectbox("Language", options=LANGUAGES.keys(), key=f"lang_{page_key}")
        selected_lang_code = LANGUAGES[selected_lang_name]

        if st.button("End Conversation & Clear History", key=f"clear_{page_key}"):
            chat_key = f'chat_history_{page_key}'
            if chat_key in st.session_state: st.session_state[chat_key] = []
            st.rerun() # Rerun needed here to clear the display

        st.header("Audio Input")
        st.write("Click to start, click again to stop.")
        audio_bytes = audio_recorder(
            text="", recording_color="#e84040", neutral_color="#6aa36f",
            icon_name="microphone", pause_threshold=300.0, key=f"recorder_{page_key}"
        )
    return audio_bytes, selected_lang_code

def handle_audio_processing(agent_runner, audio_bytes, selected_lang_code, page_key=""):
    """Core logic loop, using page-specific chat history key."""
    chat_key = f'chat_history_{page_key}'
    if chat_key not in st.session_state: st.session_state[chat_key] = []

    # Display user audio (but don't save to history)
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.audio(audio_bytes, format="audio/wav")

    # Transcribe
    text_query = call_higgs_stt(audio_bytes, selected_lang_code)

    if text_query:
        # Add transcribed text to history and display it
        timestamp = datetime.now().isoformat()
        user_message = {"role": "Doctor (transcribed)", "content": text_query, "avatar": "🧑‍⚕️", "timestamp": timestamp}
        st.session_state[chat_key].append(user_message)
        with st.chat_message(user_message["role"], avatar=user_message.get("avatar")):
            st.write(user_message["content"])

        # Run the appropriate agent logic
        with st.spinner("Thinking..."):
            answer_with_emotion, raw_transcript = agent_runner(text_query) # This calls the specific tool

        # Process the agent's response
        if answer_with_emotion:
            # Generate speech (TTS)
            audio_response = call_higgs_tts(answer_with_emotion, selected_lang_code)

            # Clean the answer for display
            clean_answer = re.sub(r'\[emotion: \w+\]', '', answer_with_emotion).strip()
            timestamp = datetime.now().isoformat()
            assistant_message = {
                "role": "assistant", "content": clean_answer, "avatar": "🤖",
                "transcript": raw_transcript, "timestamp": timestamp
            }
            st.session_state[chat_key].append(assistant_message)

            # Display the assistant's response and potentially the playback button
            # This needs to happen *outside* the initial transcription display block
            # Rerunning will handle drawing the new message from session state
            # but we need to ensure the audio plays *before* the rerun
            if audio_response:
                 st.session_state[f'autoplay_{timestamp}'] = audio_response # Store audio bytes for autoplay after rerun

            # Use st.rerun() to redraw the chat including the new message
            # and trigger autoplay if needed
            # st.rerun()

# Add a check at the top of app.py and page scripts for autoplay:
def check_autoplay():
    """Checks session state for audio to autoplay and plays it."""
    played_keys = []
    for key, audio_bytes in st.session_state.items():
        if key.startswith('autoplay_'):
            st.audio(audio_bytes, autoplay=True)
            played_keys.append(key)
    # Clean up played audio keys
    for key in played_keys:
        del st.session_state[key]

# Call check_autoplay() at the beginning of app.py and page scripts, right after imports/config