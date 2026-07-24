"""
Streamlit rewrite of the RAG-Chatbot UI (sidebar with real conversation history +
welcome screen + suggestion cards + sourced message bubbles), for deployment on
Streamlit Community Cloud. Reuses the existing RAG logic (app/rag.py) directly -
no FastAPI needed here.
"""
import asyncio
import base64
import hashlib
import io
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

import edge_tts
import streamlit as st
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    from streamlit_mic_recorder import mic_recorder
except Exception:  # noqa: BLE001  # pragma: no cover - package is installed in deployed app runtime
    mic_recorder = None

# Bridge Streamlit's secrets manager into environment variables so app/config.py
# (which reads via os.getenv) keeps working unchanged, whether run locally with .env
# or deployed with Streamlit Cloud's Secrets.
#
# Only touch st.secrets if a secrets file actually exists - accessing it when one
# doesn't (e.g. running locally with just a .env file) makes Streamlit render a
# "No secrets found" message directly on the page, which a plain try/except can't
# suppress since Streamlit displays it as a side effect, not just an exception.
_secrets_file_candidates = [
    Path.home() / ".streamlit" / "secrets.toml",
    Path(".streamlit") / "secrets.toml",
]
if any(p.exists() for p in _secrets_file_candidates):
    for _key in ("GROQ_API_KEY", "GROQ_MODEL", "QDRANT_URL", "QDRANT_API_KEY",
                 "QDRANT_COLLECTION", "EMBEDDING_MODEL"):
        try:
            if _key in st.secrets:
                os.environ[_key] = str(st.secrets[_key])
        except Exception:  # noqa: BLE001, S110
            pass

from app import auth
from app.rag import answer_question, upsert_documents

TTS_VOICE = "en-US-AriaNeural"


async def _generate_speech(text: str, voice: str = TTS_VOICE) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data


def text_to_speech_bytes(text: str, voice: str = TTS_VOICE, timeout: float = 15.0) -> bytes:
    return asyncio.run(asyncio.wait_for(_generate_speech(text, voice), timeout=timeout))


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Transcribe raw browser-captured audio bytes using Groq Whisper.
    Returns an empty string when the audio payload is empty or the API key is unavailable.
    """
    if not audio_bytes:
        return ""

    api_key = os.getenv("GROQ_API_KEY") or ""
    if not api_key:
        return ""

    client = Groq(api_key=api_key)
    transcription = client.audio.transcriptions.create(
        file=("recorded_audio.wav", audio_bytes),
        model="whisper-large-v3",
    )
    return transcription.text.strip()


st.set_page_config(page_title="RAG-Chatbot", page_icon="◆", layout="wide")

SUGGESTIONS = [
    ("🧠", "What is RAG?", "Explain Retrieval-Augmented Generation"),
    ("🔍", "Tell me about NETSOL", "Search company overview and products"),
    ("🗄️", "What is a vector database?", "Understand high-dimensional embeddings"),
    ("💡", "Explain photosynthesis", "Learn about energy conversion in plants"),
]

# ---------- theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ═══ Ethereal Glass palette (electric teal + violet on deep navy) ═══ */
:root {
    --bg-primary: #0a0d1c;
    --bg-secondary: #0e1120;
    --bg-tertiary: #14172a;
    --bg-hover: #1b1f36;
    --border: rgba(255,255,255,0.08);
    --border-strong: rgba(255,255,255,0.14);
    --glass: rgba(255,255,255,0.06);
    --glass-hover: rgba(255,255,255,0.10);
    --text-primary: #e8e6f0;
    --text-secondary: #a8a4ba;
    --text-muted: #6f6b82;
    --teal: #00d4aa;
    --teal-dim: #14b896;
    --violet: #7c6ef0;
    --coral: #ff6b8a;
    --teal-glow: rgba(0,212,170,0.35);
    --teal-soft: rgba(0,212,170,0.10);
    --accent: #00d4aa;
    --accent-glow: rgba(0,212,170,0.25);
}

#MainMenu, footer, header { visibility: hidden; }

.stApp { background: var(--bg-primary); color: var(--text-primary); font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: rgba(10,13,28,0.75);
    backdrop-filter: blur(24px);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stButton>button {
    background: transparent; color: var(--text-secondary); border: 1px solid transparent;
    border-radius: 10px; font-weight: 500; width: 100%; text-align: left; font-size: 13px;
    padding: 8px 12px; transition: all 0.2s;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background: var(--glass); color: var(--text-primary); border-color: var(--border);
}
/* New Chat button (first button in sidebar) gets the teal glass look */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:nth-child(2) .stButton>button {
    background: linear-gradient(135deg, rgba(0,212,170,0.16), rgba(124,110,240,0.12));
    border: 1px solid rgba(0,212,170,0.30); color: var(--teal);
    text-align: center; font-weight: 600;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:nth-child(2) .stButton>button:hover {
    background: linear-gradient(135deg, rgba(0,212,170,0.24), rgba(124,110,240,0.18)); color: var(--teal);
}

/* ═══ Header ═══ */
.rc-header {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 14px; border-bottom: 1px solid var(--border); margin-bottom: 18px;
}
.rc-dot { width: 8px; height: 8px; background: var(--teal); border-radius: 50%; box-shadow: 0 0 8px var(--teal-glow); animation: glow 2.5s ease-in-out infinite; }
.rc-brand { font-size: 15px; font-weight: 600; letter-spacing: -0.02em; }

/* ═══ Welcome ═══ */
.welcome { text-align: center; padding: 28px 24px 18px; }
.welcome h2 {
    font-size: 27px; font-weight: 700; margin-bottom: 10px; color: var(--teal);
    text-shadow: 0 0 14px rgba(0,212,170,0.4);
}
.welcome p { font-size: 14px; color: var(--text-muted); max-width: 460px; margin: 0 auto; line-height: 1.6; }

/* ═══ Suggested-prompt cards (glass) ═══ */
div[data-testid="column"] .stButton>button {
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 14px; color: var(--text-primary); text-align: left;
    padding: 14px 16px; width: 100%; font-size: 13px; font-weight: 500;
    backdrop-filter: blur(12px); transition: all 0.2s; white-space: pre-line;
}
div[data-testid="column"] .stButton>button:hover {
    border-color: rgba(0,212,170,0.35); background: var(--glass-hover); transform: translateY(-2px);
}

.message { display: flex; gap: 14px; margin-bottom: 22px; align-items: flex-start; }
.message.user { flex-direction: row-reverse; }
.avatar {
    width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600;
}
.message.user .avatar { background: var(--accent); color: #fff; }
.message.bot .avatar { background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text-secondary); }
.bubble { padding: 13px 17px; border-radius: 14px; font-size: 14px; line-height: 1.6; color: var(--text-primary); }
.message.user .bubble { background: var(--user-bubble); border: 1px solid var(--border); border-top-right-radius: 4px; }
.message.bot .bubble { background: var(--bot-bubble); border: 1px solid var(--border); border-top-left-radius: 4px; }
.bubble-time { font-size: 10.5px; color: var(--text-muted); margin-top: 8px; }

.sources { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.source-tag {
    display: inline-flex; align-items: center; padding: 5px 10px;
    background: var(--bg-primary); border: 1px solid var(--border);
    border-radius: 6px; font-size: 11.5px; color: var(--text-secondary);
}

.typing-row { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.typing-dots {
    display: flex; gap: 5px; padding: 13px 17px;
    background: var(--glass); backdrop-filter: blur(12px);
    border: 1px solid var(--border); border-radius: 16px; border-top-left-radius: 5px;
}
.typing-dots span { width: 7px; height: 7px; background: var(--teal); border-radius: 50%; animation: bounce 1.4s infinite; }
.typing-dots span:nth-child(2) { animation-delay: 0.15s; }
.typing-dots span:nth-child(3) { animation-delay: 0.3s; }
@keyframes bounce { 0%,80%,100% { opacity: 0.3; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-5px); } }
.typing-label { font-size: 12px; color: var(--text-muted); }

[data-testid="stChatInput"] textarea {
    background: var(--bg-secondary) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 12px !important;
}

/* flat icon-row action buttons under assistant messages (speak / regenerate) */
div[class*="st-key-actions_"] { margin-top: -8px; margin-bottom: 14px; }
div[class*="st-key-actions_"] .stButton { width: auto; }
div[class*="st-key-actions_"] .stButton > button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: var(--text-muted) !important; font-size: 14px !important; padding: 4px 8px !important;
    min-height: 0 !important; line-height: 1 !important;
}
div[class*="st-key-actions_"] .stButton > button:hover {
    background: var(--glass) !important; color: var(--teal) !important; border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- state: multiple conversations, each with its own history ----------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}  # id -> {"title": str, "messages": [...]}
if "current_id" not in st.session_state:
    st.session_state.current_id = None
if "pending" not in st.session_state:
    st.session_state.pending = None
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}  # "{conv_id}_{msg_index}" -> audio bytes
if "audio_errors" not in st.session_state:
    st.session_state.audio_errors = {}  # "{conv_id}_{msg_index}" -> error message
if "autoplay_key" not in st.session_state:
    st.session_state.autoplay_key = None  # the audio_key to autoplay on this render only
if "voice_prompt" not in st.session_state:
    st.session_state.voice_prompt = None
if "last_processed_mic_audio_key" not in st.session_state:
    st.session_state.last_processed_mic_audio_key = None


def new_conversation() -> str:
    conv_id = uuid.uuid4().hex[:8]
    st.session_state.conversations[conv_id] = {"title": "New chat", "messages": []}
    st.session_state.current_id = conv_id
    return conv_id


if not st.session_state.conversations:
    new_conversation()

current = st.session_state.conversations[st.session_state.current_id]


def maybe_set_title(conv: dict, first_message: str):
    if conv["title"] == "New chat":
        conv["title"] = first_message[:32] + ("..." if len(first_message) > 32 else "")


# ---------- sidebar ----------
with st.sidebar:
    st.markdown('<div style="font-size:13px;font-weight:600;color:var(--text-secondary);'
                'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">Conversations</div>',
                unsafe_allow_html=True)

    if st.button("＋ New Chat", use_container_width=True, key="new_chat_btn"):
        new_conversation()
        st.session_state.pending = None
        st.rerun()

    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)

    # most recently created first
    for conv_id in reversed(list(st.session_state.conversations.keys())):
        conv = st.session_state.conversations[conv_id]
        is_active = conv_id == st.session_state.current_id
        label = ("● " if is_active else "○ ") + conv["title"]
        if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
            st.session_state.current_id = conv_id
            st.session_state.pending = None
            st.rerun()

    # ---------- knowledge base uploader (mirrors FastAPI file upload) ----------
    st.divider()
    st.markdown(
        '<div style="font-size:13px;font-weight:600;color:var(--text-secondary);'
        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Knowledge base</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Add a document",
        type=["txt", "md", "pdf"],
        key="kb_uploader",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        sig = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("last_ingested_sig") != sig:
            with st.spinner(f"Ingesting {uploaded.name}…"):
                try:
                    n_chunks = ingest_uploaded(uploaded)
                    st.session_state.last_ingested_sig = sig
                    st.success(f"Added {n_chunks} chunk(s) from {uploaded.name}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Upload failed: {exc}")

    # ---------- voice input ----------
    if mic_recorder is not None:
        st.divider()
        st.markdown(
            '<div style="font-size:13px;font-weight:600;color:var(--text-secondary);'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Voice</div>',
            unsafe_allow_html=True,
        )
        mic_result = mic_recorder(
            start_prompt="🎙️ Speak",
            stop_prompt="⏹️ Stop",
            just_once=False,
            use_container_width=True,
            key="voice_recorder",
        )
        if isinstance(mic_result, dict):
            audio_bytes = mic_result.get("bytes") or mic_result.get("audio_bytes")
            if audio_bytes:
                audio_key = hashlib.sha1(audio_bytes).hexdigest()
                if audio_key != st.session_state.last_processed_mic_audio_key:
                    with st.spinner("Transcribing audio…"):
                        transcript = transcribe_audio_bytes(audio_bytes)
                    st.session_state.last_processed_mic_audio_key = audio_key
                    if transcript:
                        st.session_state.voice_prompt = transcript

    # ---------- account footer ----------
    st.divider()
    _uname = st.session_state.get("username") or "Guest"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:2px;">'
        f'<div style="width:30px;height:30px;border-radius:50%;background:var(--accent);'
        f'color:#fff;display:flex;align-items:center;justify-content:center;'
        f'font-size:13px;font-weight:600;flex-shrink:0;">{_uname[:1].upper()}</div>'
        f'<div><div style="font-size:13px;font-weight:500;">{_uname}</div>'
        f'<div style="font-size:11px;color:var(--text-muted);">RAG Workspace</div></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("Sign out", use_container_width=True, key="signout_btn"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

# ---------- header ----------
st.markdown(
    '<div class="rc-header"><div class="rc-dot"></div><div class="rc-brand">RAG-Chatbot</div></div>',
    unsafe_allow_html=True,
)


BOT_ICON = (
    '<svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/>'
    '<path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>'
)


def render_message(msg: dict, idx: int, is_last: bool):
    role = msg["role"]
    css_role = "user" if role == "user" else "bot"
    avatar = (st.session_state.get("username") or "U")[:1].upper() if role == "user" else BOT_ICON
    content = msg["content"].replace("\n", "<br>")

    sources_html = ""
    if role == "assistant" and msg.get("sources"):
        unique_sources = list(dict.fromkeys(s.get("source", "unknown") for s in msg["sources"]))
        tags = "".join(f'<span class="source-tag">📄 {s}</span>' for s in unique_sources)
        sources_html = f'<div class="sources">{tags}</div>'

    st.markdown(
        f'<div class="message {css_role}">'
        f'<div class="avatar">{avatar}</div>'
        f'<div class="msg-body">'
        f'<div class="bubble">{content}{sources_html}</div>'
        f'<div class="bubble-time">{msg.get("time", "")}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if role != "assistant":
        return

    audio_key = f"{st.session_state.current_id}_{idx}"

    # Play audio with zero visible UI - a hidden <audio autoplay> element, only
    # injected on the run right after it was (re)triggered, then cleared.
    if st.session_state.autoplay_key == audio_key and audio_key in st.session_state.audio_cache:
        b64 = base64.b64encode(st.session_state.audio_cache[audio_key]).decode()
        st.markdown(
            f'<audio autoplay style="display:none"><source src="data:audio/mp3;base64,{b64}"></audio>',
            unsafe_allow_html=True,
        )
        st.session_state.autoplay_key = None

    with st.container(key=f"actions_{audio_key}"):
        n_cols = 2 if is_last else 1
        cols = st.columns([1] * n_cols + [20])
        with cols[0]:
            if st.button("🔊", key=f"tts_{audio_key}", help="Read aloud"):
                st.session_state.audio_errors.pop(audio_key, None)
                if audio_key not in st.session_state.audio_cache:
                    try:
                        st.session_state.audio_cache[audio_key] = text_to_speech_bytes(msg["content"])
                    except Exception as e:  # noqa: BLE001
                        st.session_state.audio_errors[audio_key] = str(e)
                if audio_key in st.session_state.audio_cache:
                    st.session_state.autoplay_key = audio_key
                st.rerun()
        if is_last:
            with cols[1]:
                if st.button("🔄", key=f"regen_{audio_key}", help="Regenerate response"):
                    last_user = next(
                        (m["content"] for m in reversed(current["messages"]) if m["role"] == "user"), None
                    )
                    if last_user:
                        current["messages"] = current["messages"][:-2]
                        current["messages"].append(
                            {"role": "user", "content": last_user, "time": datetime.now(timezone.utc).strftime("%H:%M")}
                        )
                        st.session_state.pending = last_user
                        st.rerun()

    if audio_key in st.session_state.audio_errors:
        st.error(f"Couldn't generate audio: {st.session_state.audio_errors[audio_key]}")


chat_area = st.container()

with chat_area:
    if not current["messages"]:
        st.markdown('<div class="welcome">', unsafe_allow_html=True)
        st.markdown("<h2>How can I help you today?</h2>", unsafe_allow_html=True)
        st.markdown(
            '<div class="welcome">'
            '<div class="bot-hero"><div class="bot-node"><span class="pulse-ring"></span>'
            f"{BOT_ICON}</div></div>"
            "<h2>Ask anything. Know everything.</h2>"
            f"<p>{_sub}</p></div>",
            unsafe_allow_html=True,
        )

        cols = st.columns(2)
        for i, (icon, title, desc) in enumerate(SUGGESTIONS):
            with cols[i % 2]:
                if st.button(f"{icon}  {title}\n\n{desc}", key=f"sugg_{i}", use_container_width=True):
                    current["messages"].append(
                        {"role": "user", "content": title, "time": datetime.now(timezone.utc).strftime("%H:%M")}
                    )
                    maybe_set_title(current, title)
                    st.session_state.pending = title
                    st.rerun()
    else:
        last_idx = len(current["messages"]) - 1
        for idx, msg in enumerate(current["messages"]):
            render_message(msg, idx, is_last=(idx == last_idx and not st.session_state.pending))

    if st.session_state.pending:
        st.markdown(
            '<div class="typing-row"><div class="avatar" style="background:var(--bg-tertiary);'
            'border:1px solid var(--border);color:var(--text-secondary);">R</div>'
            '<div class="typing-dots"><span></span><span></span><span></span></div>'
            '<span class="typing-label">Searching knowledge base…</span></div>',
            unsafe_allow_html=True,
        )
        try:
            result = answer_question(
                st.session_state.pending,
                history=[{"role": m["role"], "content": m["content"]} for m in current["messages"][:-1]],
            )
            answer, sources = result["answer"], result["sources"]
        except Exception as e:  # noqa: BLE001
            answer, sources = f"Something went wrong: {e}", []

        current["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "time": datetime.now(timezone.utc).strftime("%H:%M"),
        })
        st.session_state.pending = None
        st.rerun()

# Top-level chat_input docks to the bottom of the page (voice input lives in the sidebar).
prompt = st.chat_input("Ask anything...")

if st.session_state.voice_prompt:
    st.info(f"Voice transcript: {st.session_state.voice_prompt}")

final_prompt = (prompt or st.session_state.voice_prompt or "").strip()
if final_prompt:
    current["messages"].append(
        {"role": "user", "content": final_prompt, "time": datetime.now(timezone.utc).strftime("%H:%M")}
    )
    maybe_set_title(current, final_prompt)
    st.session_state.pending = final_prompt
    st.session_state.voice_prompt = None
    st.rerun()