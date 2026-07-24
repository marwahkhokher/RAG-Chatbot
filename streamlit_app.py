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
    ("What is RAG?", "Explain Retrieval-Augmented Generation"),
    ("Tell me about NETSOL", "Search company overview and products"),
    ("What is a vector database?", "Understand high-dimensional embeddings"),
    ("Explain photosynthesis", "Learn about energy conversion in plants"),
]

# ---------- theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0d0d0f;
    --bg-secondary: #16161a;
    --bg-tertiary: #1e1e24;
    --bg-hover: #2a2a32;
    --bg-active: #32323c;
    --border: #2a2a32;
    --text-primary: #f0f0f5;
    --text-secondary: #a0a0ad;
    --text-muted: #666670;
    --accent: #5b7cff;
    --accent-glow: rgba(91, 124, 255, 0.15);
    --user-bubble: #2e2e36;
    --bot-bubble: #18181e;
}

#MainMenu, footer, header { visibility: hidden; }

.stApp { background: var(--bg-primary); color: var(--text-primary); font-family: 'Inter', sans-serif; }

/* Center the chat + input in a fixed-width column like the FastAPI app */
.block-container {
    max-width: 820px !important;
    margin: 0 auto !important;
    padding-top: 2rem !important;
    padding-bottom: 6rem !important;
}

section[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stButton>button {
    background: transparent; color: var(--text-secondary); border: 1px solid transparent;
    border-radius: 8px; font-weight: 500; width: 100%; text-align: left; font-size: 13px;
    padding: 8px 12px;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background: var(--bg-hover); color: var(--text-primary);
}
/* the New Chat button specifically (first button in sidebar) gets the accent look */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:nth-child(2) .stButton>button {
    background: var(--accent); color: #fff; text-align: center; font-weight: 600;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:nth-child(2) .stButton>button:hover {
    background: #7b98ff;
}

.rc-header {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 14px; border-bottom: 1px solid var(--border); margin-bottom: 18px;
}
.rc-dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; animation: glow 2s ease-in-out infinite; }
@keyframes glow { 0%,100% { box-shadow: 0 0 0 0 var(--accent-glow); } 50% { box-shadow: 0 0 8px 3px var(--accent-glow); } }
.rc-brand { font-size: 15px; font-weight: 600; letter-spacing: -0.02em; }

.welcome { text-align: center; padding: 40px 24px 20px; }
.welcome h2 { font-size: 20px; font-weight: 600; margin-bottom: 8px; }
.welcome p { font-size: 14px; color: var(--text-muted); max-width: 440px; margin: 0 auto; line-height: 1.6; }

div[data-testid="column"] .stButton>button {
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 10px; color: var(--text-primary); text-align: left;
    padding: 12px 14px; width: 100%; font-size: 13px; font-weight: 500;
}
div[data-testid="column"] .stButton>button:hover { border-color: var(--accent); background: var(--bg-tertiary); }

.message { display: flex; gap: 14px; margin-bottom: 22px; align-items: flex-start; }
.message.user { flex-direction: row-reverse; }
.avatar {
    width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600;
}
.message.user .avatar { background: linear-gradient(135deg, var(--accent), #4361ee); color: #fff; }
.message.bot .avatar { background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--accent); }
.avatar svg { width: 16px; height: 16px; stroke: var(--accent); fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
.msg-body { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.bubble { padding: 14px 18px; border-radius: 14px; font-size: 14px; line-height: 1.7; color: var(--text-primary); max-width: 88%; overflow-wrap: anywhere; }
.message.user .bubble { background: var(--bg-tertiary); border: 1px solid var(--border); border-top-right-radius: 4px; margin-left: auto; }
.message.bot .bubble { background: transparent; padding: 4px 0 0; }
.bubble-time { font-size: 10.5px; color: var(--text-muted); margin-top: 6px; }
.message.user .bubble-time { text-align: right; }

.sources { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.source-tag {
    display: inline-flex; align-items: center; gap: 5px; padding: 4px 11px;
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 999px; font-size: 11.5px; color: var(--text-secondary);
}

.typing-row { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
.typing-dots { display: flex; gap: 5px; padding: 13px 17px; background: var(--bot-bubble); border: 1px solid var(--border); border-radius: 14px; border-top-left-radius: 4px; }
.typing-dots span { width: 7px; height: 7px; background: var(--text-muted); border-radius: 50%; animation: bounce 1.4s infinite; }
.typing-dots span:nth-child(2) { animation-delay: 0.15s; }
.typing-dots span:nth-child(3) { animation-delay: 0.3s; }
@keyframes bounce { 0%,80%,100% { opacity: 0.25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-4px); } }
.typing-label { font-size: 12px; color: var(--text-muted); }

[data-testid="stChatInput"] { background: transparent !important; }
[data-testid="stChatInput"] > div {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 26px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important; border: none !important;
    color: var(--text-primary) !important; font-size: 14px !important;
}
[data-testid="stChatInput"]:focus-within > div {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
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
    background: var(--bg-hover) !important; color: var(--text-primary) !important; border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------- knowledge-base ingestion (mirrors the FastAPI /upload endpoint) ----------
def ingest_uploaded(uploaded) -> int:
    """Ingest a Streamlit UploadedFile (.txt/.md/.pdf) into Qdrant. Returns #chunks."""
    raw = uploaded.getvalue()
    suffix = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
    if suffix == "pdf":
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = raw.decode("utf-8", errors="replace")

    if not text.strip():
        raise ValueError("File appears to be empty.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    upsert_documents(chunks, [uploaded.name] * len(chunks))
    return len(chunks)


# ---------- face-recognition auth gate (mirrors the FastAPI auth overlay) ----------
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("username", "")


def render_auth_gate() -> None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            '<div style="text-align:center;padding-top:8px;">'
            '<div style="font-size:22px;font-weight:600;letter-spacing:-0.02em;">RAG-Chatbot</div>'
            '<div style="color:var(--text-muted);font-size:14px;margin-top:4px;">'
            'Sign in with your face to continue</div></div>',
            unsafe_allow_html=True,
        )

        if not auth.FACE_LIB_AVAILABLE:
            st.info("Face recognition isn't available in this environment. You can continue as a guest.")

        mode = st.radio(
            "Mode",
            ["Sign In", "Register"],
            horizontal=True,
            label_visibility="collapsed",
            disabled=not auth.FACE_LIB_AVAILABLE,
        )

        reg_name = ""
        if mode == "Register":
            reg_name = st.text_input("Your name", placeholder="Enter your name")

        img = st.camera_input("Look at the camera") if auth.FACE_LIB_AVAILABLE else None

        c1, c2 = st.columns(2)
        with c1:
            act = st.button(
                "Register Face" if mode == "Register" else "Sign In",
                type="primary",
                use_container_width=True,
                disabled=not auth.FACE_LIB_AVAILABLE,
            )
        with c2:
            skip = st.button("Continue as Guest", use_container_width=True)

        if skip:
            st.session_state.authenticated = True
            st.session_state.username = "Guest"
            st.rerun()

        if act:
            if img is None:
                st.error("Please capture a photo first using the camera above.")
                return
            if mode == "Register" and not reg_name.strip():
                st.error("Please enter your name to register.")
                return
            try:
                arr = auth._bytes_to_image(img.getvalue())
                result = (
                    auth.register_face(reg_name, arr)
                    if mode == "Register"
                    else auth.login_face(arr)
                )
            except RuntimeError as exc:
                st.error(str(exc))
                return

            if result["authenticated"]:
                st.session_state.authenticated = True
                st.session_state.username = result["username"]
                st.rerun()
            else:
                st.error(result["message"])


if not st.session_state.authenticated:
    render_auth_gate()
    st.stop()


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
        _u = st.session_state.get("username") or ""
        _greet = f"Hello, {_u}! How can I help?" if _u and _u != "Guest" else "How can I help you today?"
        st.markdown(f"<h2>{_greet}</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p>Ask me anything. I'll search your knowledge base and provide accurate, sourced answers.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        cols = st.columns(2)
        for i, (title, desc) in enumerate(SUGGESTIONS):
            with cols[i % 2]:
                if st.button(f"{title}\n\n{desc}", key=f"sugg_{i}", use_container_width=True):
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
            f'border:1px solid var(--border);">{BOT_ICON}</div>'
            '<div class="typing-dots"><span></span><span></span><span></span></div>'
            '<span class="typing-label">Searching knowledge base...</span></div>',
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