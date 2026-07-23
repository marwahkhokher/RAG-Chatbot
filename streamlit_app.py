"""
Streamlit rewrite of the RAG-Chatbot UI (sidebar with real conversation history +
welcome screen + suggestion cards + sourced message bubbles), for deployment on
Streamlit Community Cloud. Reuses the existing RAG logic (app/rag.py) directly -
no FastAPI needed here.
"""
import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

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
        except Exception:
            pass

from app.rag import answer_question  # noqa: E402

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

.message { display: flex; gap: 12px; max-width: 78%; margin-bottom: 16px; }
.message.user { margin-left: auto; flex-direction: row-reverse; }
.message.bot { margin-right: auto; }
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

.sources { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.source-tag {
    display: inline-flex; align-items: center; padding: 5px 10px;
    background: var(--bg-primary); border: 1px solid var(--border);
    border-radius: 6px; font-size: 11.5px; color: var(--text-secondary);
}

.typing-row { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
.typing-dots { display: flex; gap: 5px; padding: 13px 17px; background: var(--bot-bubble); border: 1px solid var(--border); border-radius: 14px; border-top-left-radius: 4px; }
.typing-dots span { width: 7px; height: 7px; background: var(--text-muted); border-radius: 50%; animation: bounce 1.4s infinite; }
.typing-dots span:nth-child(2) { animation-delay: 0.15s; }
.typing-dots span:nth-child(3) { animation-delay: 0.3s; }
@keyframes bounce { 0%,80%,100% { opacity: 0.25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-4px); } }
.typing-label { font-size: 12px; color: var(--text-muted); }

[data-testid="stChatInput"] textarea {
    background: var(--bg-secondary) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 12px !important;
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

# ---------- header ----------
st.markdown(
    '<div class="rc-header"><div class="rc-dot"></div><div class="rc-brand">RAG-Chatbot</div></div>',
    unsafe_allow_html=True,
)


def render_message(msg: dict):
    role = msg["role"]
    css_role = "user" if role == "user" else "bot"
    avatar = "U" if role == "user" else "R"
    content = msg["content"].replace("\n", "<br>")

    sources_html = ""
    if role == "assistant" and msg.get("sources"):
        unique_sources = list(dict.fromkeys(s.get("source", "unknown") for s in msg["sources"]))
        tags = "".join(f'<span class="source-tag">{s}</span>' for s in unique_sources)
        sources_html = f'<div class="sources">{tags}</div>'

    st.markdown(
        f'<div class="message {css_role}"><div class="avatar">{avatar}</div>'
        f'<div class="bubble">{content}{sources_html}'
        f'<div class="bubble-time">{msg.get("time", "")}</div></div></div>',
        unsafe_allow_html=True,
    )


chat_area = st.container()

with chat_area:
    if not current["messages"]:
        st.markdown('<div class="welcome">', unsafe_allow_html=True)
        st.markdown("<h2>How can I help you today?</h2>", unsafe_allow_html=True)
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
                        {"role": "user", "content": title, "time": datetime.now().strftime("%H:%M")}
                    )
                    maybe_set_title(current, title)
                    st.session_state.pending = title
                    st.rerun()
    else:
        for msg in current["messages"]:
            render_message(msg)

    if st.session_state.pending:
        st.markdown(
            '<div class="typing-row"><div class="avatar" style="background:var(--bg-tertiary);'
            'border:1px solid var(--border);color:var(--text-secondary);">R</div>'
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
        except Exception as e:
            answer, sources = f"Something went wrong: {e}", []

        current["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "time": datetime.now().strftime("%H:%M"),
        })
        st.session_state.pending = None
        st.rerun()

    # Regenerate button under the last bot response
    if (
        current["messages"]
        and current["messages"][-1]["role"] == "assistant"
        and not st.session_state.pending
    ):
        last_user = next(
            (m["content"] for m in reversed(current["messages"]) if m["role"] == "user"), None
        )
        if last_user and st.button("🔄 Regenerate response"):
            current["messages"] = current["messages"][:-2]
            current["messages"].append(
                {"role": "user", "content": last_user, "time": datetime.now().strftime("%H:%M")}
            )
            st.session_state.pending = last_user
            st.rerun()

prompt = st.chat_input("Ask anything...")
if prompt:
    current["messages"].append(
        {"role": "user", "content": prompt, "time": datetime.now().strftime("%H:%M")}
    )
    maybe_set_title(current, prompt)
    st.session_state.pending = prompt
    st.rerun()