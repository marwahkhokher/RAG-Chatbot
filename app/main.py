from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.rag import answer_question

app = FastAPI(title="RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    try:
        history = [m.model_dump() for m in req.history]
        result = answer_question(req.question, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>RAG-Chatbot — ask me anything</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-0: #050f0e;
    --bg-1: #0a1e1c;
    --panel: rgba(15, 41, 39, 0.55);
    --panel-border: rgba(0, 229, 192, 0.16);
    --cyan: #00e5c0;
    --cyan-dim: rgba(0, 229, 192, 0.35);
    --amber: #ffb454;
    --amber-dim: rgba(255, 180, 84, 0.35);
    --text: #eaf7f4;
    --muted: #6f938f;
    --danger: #ff6b6b;
  }

  * { box-sizing: border-box; }

  html, body {
    margin: 0;
    height: 100%;
    background: var(--bg-0);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    overflow: hidden;
  }

  /* ambient background: soft grid + drifting glow blobs */
  .field {
    position: fixed;
    inset: 0;
    z-index: 0;
    background:
      radial-gradient(ellipse 60% 40% at 20% 15%, rgba(0,229,192,0.10), transparent 60%),
      radial-gradient(ellipse 50% 35% at 85% 80%, rgba(255,180,84,0.08), transparent 60%),
      linear-gradient(var(--bg-1), var(--bg-0));
  }
  .field::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,229,192,0.045) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,229,192,0.045) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: radial-gradient(ellipse 70% 70% at 50% 40%, black 40%, transparent 90%);
  }

  .app {
    position: relative;
    z-index: 1;
    max-width: 760px;
    margin: 0 auto;
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 28px 20px 20px;
  }

  header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--panel-border);
    margin-bottom: 16px;
  }

  .ping {
    position: relative;
    width: 40px;
    height: 40px;
    flex-shrink: 0;
  }
  .ping-core {
    position: absolute;
    inset: 13px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 12px 2px var(--cyan-dim);
  }
  .ping-ring {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 1px solid var(--cyan);
    opacity: 0;
    animation: pingwave 2.8s ease-out infinite;
  }
  .ping-ring:nth-child(2) { animation-delay: 0.9s; }
  .ping-ring:nth-child(3) { animation-delay: 1.8s; }
  @keyframes pingwave {
    0%   { transform: scale(0.35); opacity: 0.65; }
    100% { transform: scale(1.5); opacity: 0; }
  }

  .brand h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin: 0;
  }
  .brand p {
    margin: 2px 0 0;
    font-size: 12.5px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
  }

  .status {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan);
  }

  #chat {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 6px 4px 20px;
  }
  #chat::-webkit-scrollbar { width: 6px; }
  #chat::-webkit-scrollbar-thumb { background: var(--panel-border); border-radius: 3px; }

  .empty-state {
    margin: auto;
    text-align: center;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.7;
  }
  .empty-state .big {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 17px;
    color: var(--text);
    margin-bottom: 6px;
  }

  .row { display: flex; gap: 10px; max-width: 88%; animation: rise 0.28s ease; }
  .row.user { align-self: flex-end; flex-direction: row-reverse; }
  @keyframes rise {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .avatar {
    width: 26px; height: 26px;
    border-radius: 7px;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    margin-top: 2px;
  }
  .row.bot .avatar { background: rgba(0,229,192,0.14); color: var(--cyan); border: 1px solid var(--cyan-dim); }
  .row.user .avatar { background: rgba(255,180,84,0.14); color: var(--amber); border: 1px solid var(--amber-dim); }

  .bubble {
    padding: 11px 15px;
    border-radius: 12px;
    font-size: 14.5px;
    line-height: 1.55;
    background: var(--panel);
    border: 1px solid var(--panel-border);
    backdrop-filter: blur(6px);
  }
  .row.bot .bubble { border-top-left-radius: 3px; }
  .row.user .bubble { border-top-right-radius: 3px; background: rgba(255,180,84,0.08); border-color: var(--amber-dim); }

  .typing { display: flex; gap: 4px; padding: 3px 2px; }
  .typing span {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--cyan);
    animation: bob 1.1s infinite ease-in-out;
  }
  .typing span:nth-child(2) { animation-delay: 0.15s; }
  .typing span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes bob {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30% { transform: translateY(-4px); opacity: 1; }
  }

  form {
    display: flex;
    gap: 10px;
    padding-top: 14px;
  }
  #question {
    flex: 1;
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 13px 16px;
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 14.5px;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  #question::placeholder { color: var(--muted); }
  #question:focus {
    border-color: var(--cyan);
    box-shadow: 0 0 0 3px rgba(0,229,192,0.12);
  }

  button.send {
    background: var(--cyan);
    color: #04211e;
    border: none;
    border-radius: 10px;
    padding: 0 22px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: transform 0.1s, box-shadow 0.15s;
  }
  button.send:hover { box-shadow: 0 0 16px var(--cyan-dim); }
  button.send:active { transform: scale(0.96); }
  button.send:disabled { opacity: 0.5; cursor: default; box-shadow: none; }

  @media (prefers-reduced-motion: reduce) {
    .ping-ring, .typing span { animation: none; }
    .row { animation: none; }
  }
</style>
</head>
<body>
  <div class="field"></div>

  <div class="app">
    <header>
      <div class="ping">
        <div class="ping-ring"></div>
        <div class="ping-ring"></div>
        <div class="ping-ring"></div>
        <div class="ping-core"></div>
      </div>
      <div class="brand">
        <h1>RAG-Chatbot</h1>
        <p>retrieval-augmented / ask anything</p>
      </div>
      <div class="status"><span class="status-dot"></span>online</div>
    </header>

    <div id="chat">
      <div class="empty-state" id="empty">
        <div class="big">◉ Signal ready</div>
        say hello, or ask something from the knowledge base
      </div>
    </div>

    <form id="chat-form">
      <input id="question" placeholder="Type a message..." autocomplete="off" />
      <button class="send" type="submit">SEND</button>
    </form>
  </div>

<script>
  const chat = document.getElementById('chat');
  const empty = document.getElementById('empty');
  const form = document.getElementById('chat-form');
  const input = document.getElementById('question');
  const sendBtn = form.querySelector('button');
  let history = [];

  function addRow(role, html) {
    if (empty) { empty.remove(); }
    const row = document.createElement('div');
    row.className = 'row ' + role;
    row.innerHTML = `
      <div class="avatar">${role === 'user' ? 'YOU' : '◉'}</div>
      <div class="bubble">${html}</div>
    `;
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
    return row;
  }

  function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;

    addRow('user', escapeHtml(q));
    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;

    const typingRow = addRow('bot', '<div class="typing"><span></span><span></span><span></span></div>');

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, history: history })
      });
      const data = await res.json();
      const answer = data.answer || data.detail || 'Something went wrong.';
      typingRow.querySelector('.bubble').innerHTML = escapeHtml(answer);

      if (data.answer) {
        history.push({ role: 'user', content: q });
        history.push({ role: 'assistant', content: data.answer });
      }
    } catch (err) {
      typingRow.querySelector('.bubble').innerHTML = 'Connection lost. Is the server still running?';
    } finally {
      input.disabled = false;
      sendBtn.disabled = false;
      input.focus();
    }
  });
</script>
</body>
</html>
"""