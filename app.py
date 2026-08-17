import streamlit as st
import re, json, os
import importlib
import socratic_fsm
importlib.reload(socratic_fsm)
from socratic_fsm import workflow
from langchain_core.messages import HumanMessage, AIMessage

# --- Load Dynamic Course Spec ---
SPEC_PATH = "course_spec.json"
if os.path.exists(SPEC_PATH):
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        COURSE_SPEC = json.load(f)
else:
    COURSE_SPEC = {
        "course_title": "Socratic Learning Assistant",
        "level": "GCSE",
        "target_turns": 5,
        "topics": {"General": ["General Practice"]}
    }

COURSE_TITLE = COURSE_SPEC.get("course_title", "Socratic Coach")
LEVEL = COURSE_SPEC.get("level", "GCSE/A-Level")
TARGET_TURNS = COURSE_SPEC.get("target_turns", 5)
TOPICS = COURSE_SPEC.get("topics", {})

# --- Helpers ---
def extract_clean_text(response) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return extract_clean_text(response.content)
    if isinstance(response, list) and len(response) > 0:
        first_item = response[0]
        if isinstance(first_item, dict):
            return first_item.get("text", str(first_item))
        elif hasattr(first_item, "text"):
            return first_item.text
        return extract_clean_text(first_item)
    if isinstance(response, dict):
        if "text" in response:
            return response["text"]
        elif "content" in response:
            return extract_clean_text(response["content"])
    return str(response)

def clean_latex(text: str) -> str:
    text = re.sub(r'\$([^\$]+)\$', r'<b>\1</b>', text)
    return text.replace('$', '')

def md_to_html(text: str) -> str:
    text = re.sub(r'^####\s+(.*$)', r'<h4 style="margin: 4px 0 1px 0; font-size: 1.05em; color: inherit;">\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*$)', r'<h3 style="margin: 6px 0 2px 0; font-size: 1.1em; color: inherit;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*$)', r'<h2 style="margin: 8px 0 2px 0; font-size: 1.2em; color: inherit;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'^\s*[-*]\s+(.*$)', r'<div style="margin: 1px 0;">• \1</div>', text, flags=re.MULTILINE)
    text = re.sub(r'(</(div|h2|h3|h4)>)\s*\n+', r'\1', text)
    text = re.sub(r'\n{2,}', '<br>', text)
    text = text.replace('\n', '<br>')
    return re.sub(r'(<br\s*/?>\s*)+', '<br>', text)

# --- CSS Styling ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }
    div[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    div[data-testid="stProgress"] > div > div > div { background-color: #22c55e !important; }
    h1, h2, h3 { color: #0f172a; font-family: 'Inter', sans-serif; font-weight: 700; }
    .chat-header { 
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%); 
        color: white; padding: 22px; font-weight: 700; text-align: center; 
        font-size: 1.3em; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.25);
        margin-bottom: 24px;
    }
    .selection-card {
        background: #ffffff; padding: 24px; border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; margin-bottom: 20px;
    }
    .tutor-msg { 
        background-color: #ffffff; color: #1e293b; padding: 14px 18px; 
        border-radius: 18px 18px 18px 4px; margin-bottom: 12px; max-width: 82%; 
        line-height: 1.35; border: 1px solid #e2e8f0;
    }
    .student-msg { 
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%); 
        color: white; padding: 14px 18px; border-radius: 18px 18px 4px 18px; 
        margin-bottom: 12px; max-width: 82%; margin-left: auto; line-height: 1.35;
    }
    .summary-box { 
        background: #fefce8; border-left: 5px solid #eab308; padding: 10px 14px; 
        border-radius: 12px; color: #713f12; font-size: 0.93em; margin: 8px 0; max-width: 85%; 
    }
    .stButton > button { border-radius: 12px !important; font-weight: 600 !important; }
    </style>
""", unsafe_allow_html=True)

# --- Session State ---
if "active_unit" not in st.session_state:
    st.session_state.active_unit = None
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [], "sub_topic": None, "turn_count": 0, "is_final_turn": False
    }

def reset_session():
    st.session_state.active_unit = None
    st.session_state.active_topic = None
    st.session_state.messages = []
    st.session_state.graph_state = {
        "messages": [], "sub_topic": None, "turn_count": 0, "is_final_turn": False
    }
    st.rerun()

# --- Screen Router ---
if st.session_state.active_topic is None:
    st.markdown(f'<div class="chat-header">🎓 {COURSE_TITLE} Socratic Coach</div>', unsafe_allow_html=True)
    st.markdown('<div class="selection-card">', unsafe_allow_html=True)
    st.subheader("🎯 Select Revision Target")
    st.write("Choose a subject, topic, and subtopic to begin your practice session:")
    
    # Check if topics dictionary is 3-tier (nested dict) or 2-tier (dict of lists)
    first_value = next(iter(TOPICS.values())) if TOPICS else []
    is_3_tier = isinstance(first_value, dict)

    if is_3_tier:
        # Step 1: Subject (Biology, Chemistry, Physics)
        subjects = list(TOPICS.keys())
        selected_subject = st.selectbox("🔬 Step 1: Choose Science Subject:", options=subjects)
        
        # Step 2: Main Unit/Topic (e.g. Cell biology, Forces)
        unit_dict = TOPICS.get(selected_subject, {})
        unit_keys = list(unit_dict.keys())
        selected_unit = st.selectbox("📘 Step 2: Choose Unit / Topic:", options=unit_keys)
        
        # Step 3: Specific Subtopic (e.g. Cell structure)
        subtopic_options = unit_dict.get(selected_unit, [])
        selected_subtopic = st.selectbox("🔍 Step 3: Choose Specific Subtopic:", options=subtopic_options)
    else:
        # Fallback for 2-tier structure (Topic -> Subtopic)
        unit_keys = list(TOPICS.keys()) if TOPICS else ["General"]
        selected_unit = st.selectbox("📘 Step 1: Choose Unit / Topic:", options=unit_keys)
        
        subtopic_options = TOPICS.get(selected_unit, [])
        if not subtopic_options:
            subtopic_options = [selected_unit]
        
        selected_subtopic = st.selectbox("🔍 Step 2: Choose Specific Subtopic:", options=subtopic_options)

    st.write("")
    if st.button("🚀 Start Socratic Session", type="primary", use_container_width=True):
        st.session_state.active_unit = selected_unit
        st.session_state.active_topic = selected_subtopic
        st.session_state.graph_state["sub_topic"] = selected_subtopic
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown(f'<div class="chat-header">🎓 {COURSE_TITLE} Coach</div>', unsafe_allow_html=True)
    
    student_turns = sum(1 for m in st.session_state.messages if m.get("role") == "student")

    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        
        st.metric(label="Turn Counter", value=f"{student_turns} / {TARGET_TURNS}")
        st.progress(min(student_turns / TARGET_TURNS, 1.0))
        
        st.write("---")
        if st.button("🔄 New Session / Change Topic", use_container_width=True):
            reset_session()

    if len(st.session_state.messages) == 0:
        initial_greeting = (
            f"Welcome! We're exploring **{st.session_state.active_topic}** today. "
            f"To get started, what core concept or term in this topic would you like to review?"
        )
        st.session_state.messages.append({"role": "tutor", "content": initial_greeting, "style": "tutor-msg"})
        st.session_state.graph_state["messages"].append(AIMessage(content=initial_greeting))

    for msg in st.session_state.messages:
        html_content = md_to_html(msg["content"])
        if msg["role"] == "tutor":
            div_class = msg.get("style", "tutor-msg")
            header = "💡 <b>Summary Note</b>" if div_class == "summary-box" else "🎓 <b>Tutor</b>"
            st.markdown(f'<div class="{div_class}">{header}<br><br>{html_content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="student-msg">🎒 <b>Student</b><br><br>{html_content}</div>', unsafe_allow_html=True)

    if student_turns >= TARGET_TURNS:
        st.info(f"🎉 **Session Complete!** You completed all {TARGET_TURNS} turns of the {LEVEL} Socratic dialogue.")

    is_disabled = student_turns >= TARGET_TURNS
    placeholder = "Session complete. Select a new topic in the sidebar." if is_disabled else "Type your response here..."
    
    if user_input := st.chat_input(placeholder, disabled=is_disabled):
        st.session_state.messages.append({"role": "student", "content": user_input})
        st.session_state.graph_state["messages"].append(HumanMessage(content=user_input))
        
        current_student_turns = sum(1 for m in st.session_state.messages if m.get("role") == "student")
        st.session_state.graph_state["turn_count"] = current_student_turns
        st.session_state.graph_state["is_final_turn"] = (current_student_turns >= TARGET_TURNS)

        with st.spinner("Analyzing response and generating feedback..."):
            updated_state = workflow.invoke(st.session_state.graph_state)
        
        last_msg = updated_state["messages"][-1]
        ai_reply = clean_latex(extract_clean_text(last_msg))

        split_match = re.split(r'={3,}\s*SPLIT\s*={3,}', ai_reply, flags=re.IGNORECASE)
        if len(split_match) > 1:
            st.session_state.messages.append({"role": "tutor", "content": split_match[0].strip(), "style": "tutor-msg"})
            st.session_state.messages.append({"role": "tutor", "content": split_match[1].strip(), "style": "summary-box"})
        else:
            st.session_state.messages.append({"role": "tutor", "content": ai_reply, "style": "tutor-msg"})

        st.session_state.graph_state = updated_state
        st.rerun()