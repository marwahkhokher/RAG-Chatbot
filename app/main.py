import io

import edge_tts
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pypdf import PdfReader

from app.auth import router as auth_router
from app.rag import answer_question, upsert_documents

app = FastAPI(title="RAG Chatbot", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire face-auth routes
app.include_router(auth_router)


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []


class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-AriaNeural"


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
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        communicate = edge_tts.Communicate(req.text, req.voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return Response(content=audio_data, media_type="audio/mpeg")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):  # noqa: B008
    """Accept .txt, .md, or .pdf uploads and ingest them into the RAG knowledge base."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ("txt", "md", "pdf"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload .txt, .md, or .pdf files.",
        )

    raw = await file.read()

    if suffix == "pdf":
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = raw.decode("utf-8", errors="replace")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File appears to be empty")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    sources = [file.filename] * len(chunks)
    upsert_documents(chunks, sources)

    return {
        "status": "ok",
        "filename": file.filename,
        "chunks": len(chunks),
        "message": f"Ingested '{file.filename}' as {len(chunks)} chunks into the knowledge base.",
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG-Chatbot</title>
  <meta name="description" content="An intelligent RAG chatbot with face authentication, powered by Groq LLM and Qdrant vector search.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

    :root {
      --bg-primary: #0f0f11;
      --bg-secondary: #18181b;
      --bg-tertiary: #1e1e22;
      --bg-hover: #27272b;
      --bg-active: #2e2e34;
      --border: #27272b;
      --border-light: #353539;
      --text-primary: #f4f4f5;
      --text-secondary: #a1a1aa;
      --text-muted: #63636e;
      --accent-1: #8b5cf6;
      --accent-2: #6366f1;
      --accent-3: #3b82f6;
      --accent-glow: rgba(139, 92, 246, 0.12);
      --accent-glow-strong: rgba(139, 92, 246, 0.25);
      --green: #22c55e;
      --red: #ef4444;
      --amber: #f59e0b;
      --radius: 20px;
      --radius-md: 14px;
      --radius-sm: 10px;
      --radius-xs: 6px;
      --transition: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    html, body {
      height: 100%;
      font-family: var(--font);
      background: var(--bg-primary);
      color: var(--text-primary);
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }

    /* ═══════════════════════════════════════
       AUTH OVERLAY
    ═══════════════════════════════════════ */
    .auth-overlay {
      position: fixed;
      inset: 0;
      z-index: 1000;
      background: var(--bg-primary);
      display: flex;
      align-items: center;
      justify-content: center;
      transition: opacity 0.5s ease, visibility 0.5s ease;
    }
    .auth-overlay.hidden {
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
    }

    .auth-card {
      width: 440px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 40px 36px 32px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 24px;
      box-shadow: 0 24px 80px rgba(0,0,0,0.5);
      animation: authSlideIn 0.5s ease-out;
    }
    @keyframes authSlideIn {
      from { opacity: 0; transform: translateY(24px) scale(0.96); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }

    .auth-logo {
      width: 56px; height: 56px;
      border-radius: 16px;
      background: linear-gradient(135deg, var(--accent-1), var(--accent-3));
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 0 32px var(--accent-glow-strong);
    }
    .auth-logo svg {
      width: 28px; height: 28px;
      stroke: #fff; fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }

    .auth-title {
      font-size: 22px; font-weight: 600;
      letter-spacing: -0.03em;
      text-align: center;
    }
    .auth-subtitle {
      font-size: 14px;
      color: var(--text-muted);
      text-align: center;
      margin-top: -12px;
      line-height: 1.5;
    }

    .webcam-container {
      width: 240px; height: 240px;
      border-radius: 50%;
      overflow: hidden;
      border: 3px solid var(--border-light);
      position: relative;
      background: var(--bg-tertiary);
      box-shadow: 0 0 0 8px var(--accent-glow);
      transition: border-color var(--transition), box-shadow var(--transition);
    }
    .webcam-container.scanning {
      border-color: var(--accent-1);
      box-shadow: 0 0 0 8px var(--accent-glow-strong);
      animation: scanPulse 1.5s ease-in-out infinite;
    }
    @keyframes scanPulse {
      0%, 100% { box-shadow: 0 0 0 8px var(--accent-glow); }
      50% { box-shadow: 0 0 0 16px var(--accent-glow-strong), 0 0 40px var(--accent-glow); }
    }

    .webcam-container video {
      width: 100%; height: 100%;
      object-fit: cover;
      transform: scaleX(-1);
    }
    .webcam-placeholder {
      width: 100%; height: 100%;
      display: flex; align-items: center; justify-content: center;
      color: var(--text-muted); font-size: 13px;
    }

    .auth-input {
      width: 100%;
      padding: 12px 16px;
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      color: var(--text-primary);
      font-family: var(--font);
      font-size: 14px;
      outline: none;
      transition: border-color var(--transition), box-shadow var(--transition);
    }
    .auth-input:focus {
      border-color: var(--accent-1);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }
    .auth-input::placeholder { color: var(--text-muted); }

    .auth-actions {
      display: flex;
      gap: 10px;
      width: 100%;
    }

    .auth-btn {
      flex: 1;
      padding: 12px 20px;
      border-radius: var(--radius-sm);
      font-family: var(--font);
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      border: none;
      transition: background var(--transition), transform 0.1s, opacity var(--transition);
    }
    .auth-btn:active { transform: scale(0.97); }
    .auth-btn:disabled { opacity: 0.5; cursor: default; }

    .auth-btn.primary {
      background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
      color: #fff;
    }
    .auth-btn.primary:hover:not(:disabled) {
      background: linear-gradient(135deg, #9b6cf7, #7577f2);
    }
    .auth-btn.secondary {
      background: var(--bg-tertiary);
      color: var(--text-secondary);
      border: 1px solid var(--border);
    }
    .auth-btn.secondary:hover:not(:disabled) {
      background: var(--bg-hover);
      color: var(--text-primary);
    }

    .auth-toggle {
      font-size: 13px;
      color: var(--text-muted);
    }
    .auth-toggle a {
      color: var(--accent-1);
      cursor: pointer;
      text-decoration: none;
    }
    .auth-toggle a:hover { text-decoration: underline; }

    .auth-message {
      font-size: 13px;
      text-align: center;
      padding: 8px 12px;
      border-radius: var(--radius-xs);
      width: 100%;
      display: none;
    }
    .auth-message.error {
      display: block;
      background: rgba(239,68,68,0.1);
      color: var(--red);
      border: 1px solid rgba(239,68,68,0.2);
    }
    .auth-message.success {
      display: block;
      background: rgba(34,197,94,0.1);
      color: var(--green);
      border: 1px solid rgba(34,197,94,0.2);
    }

    /* ═══════════════════════════════════════
       APP SHELL
    ═══════════════════════════════════════ */
    .app {
      display: flex;
      height: 100vh;
    }

    /* ── Sidebar ── */
    .sidebar {
      width: 272px;
      background: var(--bg-secondary);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      transition: transform var(--transition), width var(--transition), opacity var(--transition);
      overflow: hidden;
    }
    .sidebar.collapsed {
      transform: translateX(-100%);
      width: 0; opacity: 0;
      pointer-events: none;
    }

    .sidebar-header {
      padding: 16px 14px 12px;
      border-bottom: 1px solid var(--border);
    }
    .sidebar-title {
      font-size: 11px; font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }

    .new-chat-btn {
      width: 100%;
      display: flex; align-items: center; gap: 8px;
      padding: 10px 14px;
      background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
      border: none; border-radius: var(--radius-sm);
      color: #fff;
      font-family: var(--font); font-size: 13px; font-weight: 500;
      cursor: pointer;
      transition: transform 0.1s, filter var(--transition);
    }
    .new-chat-btn:hover { filter: brightness(1.1); }
    .new-chat-btn:active { transform: scale(0.97); }
    .new-chat-btn svg {
      width: 15px; height: 15px;
      stroke: currentColor; fill: none;
      stroke-width: 2; stroke-linecap: round;
    }

    .chat-list {
      flex: 1; overflow-y: auto; padding: 6px 8px;
    }
    .chat-list::-webkit-scrollbar { width: 3px; }
    .chat-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    .chat-item {
      display: flex; align-items: center; gap: 10px;
      padding: 9px 12px;
      border-radius: var(--radius-xs); cursor: pointer;
      transition: background var(--transition);
      margin-bottom: 2px;
    }
    .chat-item:hover { background: var(--bg-hover); }
    .chat-item.active { background: var(--bg-active); }
    .chat-item-icon {
      width: 7px; height: 7px;
      border-radius: 50%; background: var(--text-muted);
      flex-shrink: 0;
    }
    .chat-item.active .chat-item-icon { background: var(--accent-1); }
    .chat-item-text {
      font-size: 13px; color: var(--text-secondary);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      flex: 1;
    }
    .chat-item.active .chat-item-text { color: var(--text-primary); }
    .chat-item-time {
      font-size: 10px; color: var(--text-muted); flex-shrink: 0;
    }

    .sidebar-footer {
      padding: 12px 14px;
      border-top: 1px solid var(--border);
    }
    .user-profile {
      display: flex; align-items: center; gap: 10px;
      padding: 8px 10px;
      border-radius: var(--radius-xs); cursor: pointer;
      transition: background var(--transition);
    }
    .user-profile:hover { background: var(--bg-hover); }

    .user-avatar {
      width: 32px; height: 32px;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--accent-1), var(--accent-3));
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 600; color: #fff;
    }
    .user-info { flex: 1; }
    .user-name { font-size: 13px; font-weight: 500; color: var(--text-primary); }
    .user-role { font-size: 11px; color: var(--text-muted); }

    /* ── Main ── */
    .main {
      flex: 1;
      display: flex; flex-direction: column;
      min-width: 0; position: relative;
    }

    /* Header */
    .header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 10px 20px;
      border-bottom: 1px solid var(--border);
      background: var(--bg-secondary);
      flex-shrink: 0;
    }
    .header-left {
      display: flex; align-items: center; gap: 10px;
    }

    .icon-btn {
      width: 34px; height: 34px;
      display: flex; align-items: center; justify-content: center;
      background: transparent; border: none;
      border-radius: var(--radius-xs);
      color: var(--text-secondary); cursor: pointer;
      transition: background var(--transition), color var(--transition);
    }
    .icon-btn:hover { background: var(--bg-hover); color: var(--text-primary); }
    .icon-btn svg {
      width: 17px; height: 17px;
      stroke: currentColor; fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }

    .header-brand {
      display: flex; align-items: center; gap: 9px;
    }
    .brand-dot {
      width: 8px; height: 8px;
      background: linear-gradient(135deg, var(--accent-1), var(--accent-3));
      border-radius: 50%;
      animation: brandPulse 2.5s ease-in-out infinite;
    }
    @keyframes brandPulse {
      0%, 100% { box-shadow: 0 0 0 0 transparent; }
      50% { box-shadow: 0 0 10px 3px var(--accent-glow); }
    }
    .brand-name {
      font-size: 14px; font-weight: 600;
      letter-spacing: -0.02em;
    }
    .header-right {
      display: flex; align-items: center; gap: 2px;
    }

    /* ═══════════════════════════════════════
       CHAT AREA
    ═══════════════════════════════════════ */
    .chat-area {
      flex: 1;
      overflow-y: auto;
      padding: 28px 0;
      display: flex; flex-direction: column;
      scroll-behavior: smooth;
    }
    .chat-area::-webkit-scrollbar { width: 5px; }
    .chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

    .chat-inner {
      max-width: 760px;
      width: 100%;
      margin: 0 auto;
      padding: 0 24px;
    }

    /* Welcome */
    .welcome {
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      text-align: center;
      padding: 48px 20px 32px;
      gap: 10px;
      animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(16px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .welcome-icon {
      width: 64px; height: 64px;
      border-radius: 20px;
      background: linear-gradient(135deg, var(--accent-1), var(--accent-3));
      display: flex; align-items: center; justify-content: center;
      margin-bottom: 8px;
      box-shadow: 0 8px 32px var(--accent-glow-strong);
    }
    .welcome-icon svg {
      width: 30px; height: 30px;
      stroke: #fff; fill: none;
      stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round;
    }

    .welcome h2 {
      font-size: 26px; font-weight: 600;
      letter-spacing: -0.03em;
      background: linear-gradient(135deg, var(--text-primary), var(--text-secondary));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .welcome p {
      font-size: 14px; color: var(--text-muted);
      line-height: 1.6; max-width: 420px;
    }

    .suggestions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 16px;
      max-width: 500px;
      width: 100%;
    }
    .suggestion-card {
      padding: 14px 16px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      cursor: pointer;
      text-align: left;
      transition: border-color var(--transition), background var(--transition), transform 0.15s;
    }
    .suggestion-card:hover {
      border-color: var(--accent-1);
      background: var(--bg-tertiary);
      transform: translateY(-2px);
    }
    .suggestion-card:active { transform: translateY(0); }

    .card-icon {
      width: 28px; height: 28px;
      background: var(--accent-glow);
      border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      margin-bottom: 10px;
    }
    .card-icon svg {
      width: 14px; height: 14px;
      stroke: var(--accent-1); fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }
    .card-title { font-size: 13px; font-weight: 500; color: var(--text-primary); margin-bottom: 3px; }
    .card-desc { font-size: 11.5px; color: var(--text-muted); line-height: 1.4; }

    /* Messages */
    .message {
      display: flex; gap: 14px;
      margin-bottom: 24px;
      animation: msgIn 0.3s ease-out;
    }
    @keyframes msgIn {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .message.user {
      flex-direction: row-reverse;
    }

    .msg-avatar {
      width: 32px; height: 32px;
      border-radius: 50%;
      flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 600;
    }
    .message.user .msg-avatar {
      background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
      color: #fff;
    }
    .message.bot .msg-avatar {
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      color: var(--accent-1);
    }
    .message.bot .msg-avatar svg {
      width: 16px; height: 16px;
      stroke: var(--accent-1); fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }

    .msg-body { flex: 1; min-width: 0; }

    .msg-bubble {
      padding: 14px 18px;
      border-radius: var(--radius-md);
      font-size: 14px;
      line-height: 1.7;
      color: var(--text-primary);
      max-width: 85%;
    }
    .message.user .msg-bubble {
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-top-right-radius: 4px;
      margin-left: auto;
    }
    .message.bot .msg-bubble {
      background: transparent;
      border-top-left-radius: 4px;
    }

    /* Markdown in bot messages */
    .msg-bubble p { margin-bottom: 8px; }
    .msg-bubble p:last-child { margin-bottom: 0; }
    .msg-bubble strong { font-weight: 600; }
    .msg-bubble em { font-style: italic; color: var(--text-secondary); }
    .msg-bubble code {
      background: var(--bg-active);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 13px;
      font-family: 'SF Mono', 'Fira Code', monospace;
    }
    .msg-bubble pre {
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: var(--radius-xs);
      padding: 12px 16px;
      overflow-x: auto;
      margin: 8px 0;
    }
    .msg-bubble pre code {
      background: transparent;
      padding: 0;
    }
    .msg-bubble ul, .msg-bubble ol {
      padding-left: 20px;
      margin: 6px 0;
    }
    .msg-bubble li { margin-bottom: 4px; }

    .msg-meta {
      display: flex; align-items: center; gap: 8px;
      margin-top: 6px; padding: 0 2px;
    }
    .msg-time {
      font-size: 10.5px; color: var(--text-muted);
    }
    .msg-actions {
      display: flex; gap: 1px;
      opacity: 0; transition: opacity var(--transition);
    }
    .message:hover .msg-actions { opacity: 1; }

    .msg-action {
      width: 26px; height: 26px;
      display: flex; align-items: center; justify-content: center;
      background: transparent; border: none;
      border-radius: var(--radius-xs);
      color: var(--text-muted); cursor: pointer;
      transition: background var(--transition), color var(--transition);
    }
    .msg-action:hover { background: var(--bg-hover); color: var(--text-primary); }
    .msg-action svg {
      width: 13px; height: 13px;
      stroke: currentColor; fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }

    /* Sources */
    .sources {
      display: flex; flex-wrap: wrap; gap: 6px;
      margin-top: 10px;
    }
    .source-pill {
      display: flex; align-items: center; gap: 5px;
      padding: 4px 10px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: 999px;
      font-size: 11px; color: var(--text-secondary);
      cursor: default;
      transition: border-color var(--transition);
    }
    .source-pill:hover { border-color: var(--accent-1); }
    .source-pill svg {
      width: 11px; height: 11px;
      stroke: var(--text-muted); fill: none;
      stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round;
    }

    /* Typing indicator */
    .typing {
      display: flex; align-items: center; gap: 14px;
      margin-bottom: 24px;
      animation: msgIn 0.25s ease-out;
    }
    .typing-dots {
      display: flex; gap: 5px;
      padding: 16px 20px;
    }
    .typing-dots span {
      width: 7px; height: 7px;
      background: var(--text-muted);
      border-radius: 50%;
      animation: dotBounce 1.4s infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes dotBounce {
      0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
      40% { opacity: 1; transform: translateY(-5px); }
    }
    .typing-label {
      font-size: 12px; color: var(--text-muted);
    }

    /* ═══════════════════════════════════════
       INPUT AREA (Gemini-style)
    ═══════════════════════════════════════ */
    .input-area {
      flex-shrink: 0;
      padding: 0 24px 20px;
      display: flex;
      justify-content: center;
    }

    .input-container {
      max-width: 760px;
      width: 100%;
    }

    .input-bar {
      display: flex;
      align-items: flex-end;
      gap: 0;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: 28px;
      padding: 6px 8px 6px 6px;
      transition: border-color var(--transition), box-shadow var(--transition);
    }
    .input-bar:focus-within {
      border-color: var(--accent-1);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }

    .input-bar-left {
      display: flex;
      align-items: center;
      gap: 2px;
      padding-left: 4px;
    }

    .bar-btn {
      width: 36px; height: 36px;
      display: flex; align-items: center; justify-content: center;
      background: transparent; border: none;
      border-radius: 50%;
      color: var(--text-muted); cursor: pointer;
      transition: background var(--transition), color var(--transition);
      flex-shrink: 0;
      position: relative;
    }
    .bar-btn:hover { background: var(--bg-hover); color: var(--text-secondary); }
    .bar-btn svg {
      width: 18px; height: 18px;
      stroke: currentColor; fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }
    .bar-btn.recording {
      color: var(--red);
      animation: micPulse 1s ease-in-out infinite;
    }
    @keyframes micPulse {
      0%, 100% { background: transparent; }
      50% { background: rgba(239,68,68,0.12); }
    }

    .input-bar textarea {
      flex: 1;
      background: transparent;
      border: none; outline: none; resize: none;
      padding: 8px 12px;
      font-family: var(--font);
      font-size: 14px;
      color: var(--text-primary);
      line-height: 1.5;
      max-height: 140px;
      min-height: 22px;
    }
    .input-bar textarea::placeholder { color: var(--text-muted); }

    .input-bar-right {
      display: flex;
      align-items: center;
      gap: 2px;
    }

    .send-btn {
      width: 38px; height: 38px;
      display: flex; align-items: center; justify-content: center;
      background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
      border: none; border-radius: 50%;
      cursor: pointer;
      transition: transform 0.1s, opacity var(--transition), filter var(--transition);
      flex-shrink: 0;
    }
    .send-btn:hover { filter: brightness(1.15); }
    .send-btn:active { transform: scale(0.9); }
    .send-btn:disabled { opacity: 0.3; cursor: default; filter: none; }
    .send-btn svg {
      width: 17px; height: 17px;
      fill: #fff; stroke: none;
    }

    .input-footer {
      display: flex; align-items: center; justify-content: center;
      gap: 8px; padding-top: 10px;
    }
    .input-footer span { font-size: 11px; color: var(--text-muted); }
    .input-footer .dot { width: 3px; height: 3px; background: var(--text-muted); border-radius: 50%; }

    /* Upload toast */
    .upload-toast {
      position: fixed; bottom: 100px; left: 50%;
      transform: translateX(-50%) translateY(20px);
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 10px 20px;
      font-size: 13px; color: var(--text-secondary);
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
      opacity: 0; pointer-events: none;
      transition: opacity 0.3s, transform 0.3s;
      z-index: 500;
    }
    .upload-toast.show {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }

    /* Scroll-to-bottom */
    .scroll-btn {
      position: absolute;
      bottom: 100px; right: 40px;
      width: 36px; height: 36px;
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      color: var(--text-secondary); cursor: pointer;
      opacity: 0; transform: translateY(8px);
      transition: opacity var(--transition), transform var(--transition);
      pointer-events: none; z-index: 10;
    }
    .scroll-btn.visible {
      opacity: 1; transform: translateY(0);
      pointer-events: all;
    }
    .scroll-btn:hover { background: var(--bg-hover); }
    .scroll-btn svg {
      width: 16px; height: 16px;
      stroke: currentColor; fill: none;
      stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;
    }

    /* Hidden file input */
    #fileInput { display: none; }

    /* Responsive */
    @media (max-width: 768px) {
      .sidebar { position: absolute; height: 100%; z-index: 50; }
      .chat-inner { padding: 0 14px; }
      .input-area { padding: 0 12px 16px; }
      .suggestions { grid-template-columns: 1fr; }
      .auth-card { width: 90%; padding: 28px 24px; }
      .webcam-container { width: 180px; height: 180px; }
    }
  </style>
</head>
<body>

  <!-- ═══════ AUTH OVERLAY ═══════ -->
  <div class="auth-overlay" id="authOverlay">
    <div class="auth-card">
      <div class="auth-logo">
        <svg viewBox="0 0 24 24"><path d="M12 2a5 5 0 0 1 5 5v2a5 5 0 0 1-10 0V7a5 5 0 0 1 5-5z"/><path d="M3.05 13A9 9 0 0 0 12 21a9 9 0 0 0 8.95-8"/><line x1="12" y1="21" x2="12" y2="24"/></svg>
      </div>
      <div class="auth-title" id="authTitle">Welcome Back</div>
      <div class="auth-subtitle" id="authSubtitle">Look at the camera to sign in with your face</div>

      <div class="webcam-container" id="webcamContainer">
        <div class="webcam-placeholder" id="webcamPlaceholder">Initializing camera...</div>
        <video id="webcamVideo" autoplay playsinline></video>
      </div>

      <input
        class="auth-input"
        id="regUsername"
        type="text"
        placeholder="Enter your name"
        style="display:none"
      />

      <div class="auth-message" id="authMessage"></div>

      <div class="auth-actions">
        <button class="auth-btn primary" id="authPrimaryBtn" onclick="authAction()">
          Sign In
        </button>
        <button class="auth-btn secondary" id="authCancelBtn" onclick="skipAuth()" title="Continue without face auth">
          Skip
        </button>
      </div>

      <div class="auth-toggle" id="authToggle">
        New here? <a onclick="toggleAuthMode()">Register your face</a>
      </div>
    </div>
  </div>

  <!-- ═══════ MAIN APP ═══════ -->
  <div class="app">

    <!-- Sidebar -->
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-header">
        <div class="sidebar-title">Conversations</div>
        <button class="new-chat-btn" onclick="startNewChat()">
          <svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>
          New Chat
        </button>
      </div>
      <div class="chat-list" id="chatList">
        <div class="chat-item active" onclick="selectChat(this)">
          <div class="chat-item-icon"></div>
          <span class="chat-item-text">Current Session</span>
          <span class="chat-item-time">Now</span>
        </div>
      </div>
      <div class="sidebar-footer">
        <div class="user-profile" onclick="showAuthOverlay()">
          <div class="user-avatar" id="sidebarAvatar">?</div>
          <div class="user-info">
            <div class="user-name" id="sidebarName">Guest</div>
            <div class="user-role">RAG Workspace</div>
          </div>
        </div>
      </div>
    </aside>

    <!-- Main -->
    <div class="main">

      <!-- Header -->
      <div class="header">
        <div class="header-left">
          <button class="icon-btn" onclick="toggleSidebar()" title="Toggle sidebar">
            <svg viewBox="0 0 24 24"><path d="M3 12h18M3 6h18M3 18h18"/></svg>
          </button>
          <div class="header-brand">
            <div class="brand-dot"></div>
            <span class="brand-name">RAG-Chatbot</span>
          </div>
        </div>
        <div class="header-right">
          <button class="icon-btn" title="New Chat" onclick="startNewChat()">
            <svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>
          </button>
        </div>
      </div>

      <!-- Chat Area -->
      <div class="chat-area" id="chatArea">
        <div class="chat-inner" id="chatInner">
          <div class="welcome" id="welcome">
            <div class="welcome-icon">
              <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
            </div>
            <h2 id="welcomeGreeting">Hello! How can I help?</h2>
            <p>Ask me anything — I'll search your knowledge base and provide accurate, sourced answers.</p>
            <div class="suggestions">
              <div class="suggestion-card" onclick="sendSuggestion(this)">
                <div class="card-icon"><svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></div>
                <div class="card-title">What is RAG?</div>
                <div class="card-desc">Explain Retrieval-Augmented Generation</div>
              </div>
              <div class="suggestion-card" onclick="sendSuggestion(this)">
                <div class="card-icon"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg></div>
                <div class="card-title">Tell me about NETSOL</div>
                <div class="card-desc">Search company overview and products</div>
              </div>
              <div class="suggestion-card" onclick="sendSuggestion(this)">
                <div class="card-icon"><svg viewBox="0 0 24 24"><path d="M18 20V10M12 20V4M6 20v-6"/></svg></div>
                <div class="card-title">What is a vector database?</div>
                <div class="card-desc">Understand high-dimensional embeddings</div>
              </div>
              <div class="suggestion-card" onclick="sendSuggestion(this)">
                <div class="card-icon"><svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
                <div class="card-title">Explain photosynthesis</div>
                <div class="card-desc">Learn about energy conversion in plants</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Scroll-to-bottom -->
      <button class="scroll-btn" id="scrollBtn" onclick="scrollToBottom()">
        <svg viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>
      </button>

      <!-- Input Area -->
      <div class="input-area">
        <div class="input-container">
          <div class="input-bar" id="inputBar">
            <div class="input-bar-left">
              <button class="bar-btn" id="attachBtn" onclick="document.getElementById('fileInput').click()" title="Attach file (.txt, .md, .pdf)">
                <svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>
              </button>
            </div>
            <textarea
              id="msgInput"
              rows="1"
              placeholder="Ask anything..."
              oninput="autoResize(this); updateSendBtn()"
              onkeydown="handleKey(event)"
            ></textarea>
            <div class="input-bar-right">
              <button class="bar-btn" id="micBtn" onclick="toggleMic()" title="Voice input">
                <svg viewBox="0 0 24 24"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
              </button>
              <button class="send-btn" id="sendBtn" onclick="sendMessage()" disabled title="Send message">
                <svg viewBox="0 0 24 24"><path d="M3.478 2.405a.75.75 0 0 0-.926.94l2.432 7.905H13.5a.75.75 0 0 1 0 1.5H4.984l-2.432 7.905a.75.75 0 0 0 .926.94l18.04-8.25a.75.75 0 0 0 0-1.39L3.478 2.405z"/></svg>
              </button>
            </div>
          </div>
          <input type="file" id="fileInput" accept=".txt,.md,.pdf" onchange="handleFileUpload(event)">
          <div class="input-footer">
            <span>RAG-Chatbot</span>
            <div class="dot"></div>
            <span>Grounded in your data</span>
            <div class="dot"></div>
            <span>Enter ⏎ to send</span>
          </div>
        </div>
      </div>

    </div>
  </div>

  <!-- Upload toast -->
  <div class="upload-toast" id="uploadToast"></div>

  <script>
    // ── Element refs ──
    const chatArea     = document.getElementById('chatArea');
    const chatInner    = document.getElementById('chatInner');
    const msgInput     = document.getElementById('msgInput');
    const sendBtn      = document.getElementById('sendBtn');
    const scrollBtn    = document.getElementById('scrollBtn');
    const sidebar      = document.getElementById('sidebar');
    const authOverlay  = document.getElementById('authOverlay');
    const micBtn       = document.getElementById('micBtn');

    let history          = [];
    let lastUserQuestion = '';
    let currentUser      = sessionStorage.getItem('rag_user') || '';
    let isRegisterMode   = false;
    let recognition      = null;
    let isRecording      = false;
    let webcamStream     = null;

    // ── Initialize ──
    if (currentUser) {
      authOverlay.classList.add('hidden');
      setUserUI(currentUser);
    } else {
      startWebcam();
    }

    // ═══════════════════════════════════════
    //   AUTH FUNCTIONS
    // ═══════════════════════════════════════
    async function startWebcam() {
      const video = document.getElementById('webcamVideo');
      const placeholder = document.getElementById('webcamPlaceholder');
      try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 480, height: 480 } });
        video.srcObject = webcamStream;
        video.style.display = 'block';
        placeholder.style.display = 'none';
      } catch (err) {
        placeholder.textContent = 'Camera not available';
        console.error('Webcam error:', err);
      }
    }

    function stopWebcam() {
      if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
      }
    }

    function captureFrame() {
      const video = document.getElementById('webcamVideo');
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 480;
      canvas.height = video.videoHeight || 480;
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.9);
    }

    function toggleAuthMode() {
      isRegisterMode = !isRegisterMode;
      const title    = document.getElementById('authTitle');
      const subtitle = document.getElementById('authSubtitle');
      const regInput = document.getElementById('regUsername');
      const btn      = document.getElementById('authPrimaryBtn');
      const toggle   = document.getElementById('authToggle');

      if (isRegisterMode) {
        title.textContent    = 'Create Account';
        subtitle.textContent = 'Enter your name and look at the camera';
        regInput.style.display = 'block';
        btn.textContent      = 'Register Face';
        toggle.innerHTML     = 'Already registered? <a onclick="toggleAuthMode()">Sign in</a>';
      } else {
        title.textContent    = 'Welcome Back';
        subtitle.textContent = 'Look at the camera to sign in with your face';
        regInput.style.display = 'none';
        btn.textContent      = 'Sign In';
        toggle.innerHTML     = 'New here? <a onclick="toggleAuthMode()">Register your face</a>';
      }
      hideAuthMessage();
    }

    function showAuthMessage(text, type) {
      const el = document.getElementById('authMessage');
      el.textContent = text;
      el.className = 'auth-message ' + type;
    }
    function hideAuthMessage() {
      document.getElementById('authMessage').className = 'auth-message';
    }

    async function authAction() {
      const btn = document.getElementById('authPrimaryBtn');
      const container = document.getElementById('webcamContainer');
      btn.disabled = true;
      container.classList.add('scanning');
      hideAuthMessage();

      const image = captureFrame();

      try {
        if (isRegisterMode) {
          const username = document.getElementById('regUsername').value.trim();
          if (!username) {
            showAuthMessage('Please enter your name.', 'error');
            btn.disabled = false;
            container.classList.remove('scanning');
            return;
          }
          const res = await fetch('/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, image })
          });
          const data = await res.json();
          container.classList.remove('scanning');

          if (data.authenticated) {
            showAuthMessage(data.message, 'success');
            currentUser = data.username;
            sessionStorage.setItem('rag_user', currentUser);
            setUserUI(currentUser);
            setTimeout(() => {
              stopWebcam();
              authOverlay.classList.add('hidden');
            }, 1000);
          } else {
            showAuthMessage(data.message || data.detail || 'Registration failed.', 'error');
            btn.disabled = false;
          }
        } else {
          const res = await fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image })
          });
          const data = await res.json();
          container.classList.remove('scanning');

          if (data.authenticated) {
            showAuthMessage(data.message, 'success');
            currentUser = data.username;
            sessionStorage.setItem('rag_user', currentUser);
            setUserUI(currentUser);
            setTimeout(() => {
              stopWebcam();
              authOverlay.classList.add('hidden');
            }, 1000);
          } else {
            showAuthMessage(data.message || data.detail || 'Login failed.', 'error');
            btn.disabled = false;
          }
        }
      } catch (err) {
        container.classList.remove('scanning');
        showAuthMessage('Server error. Is the backend running?', 'error');
        btn.disabled = false;
      }
    }

    function skipAuth() {
      currentUser = 'Guest';
      sessionStorage.setItem('rag_user', currentUser);
      setUserUI(currentUser);
      stopWebcam();
      authOverlay.classList.add('hidden');
    }

    function showAuthOverlay() {
      authOverlay.classList.remove('hidden');
      startWebcam();
    }

    function setUserUI(name) {
      document.getElementById('sidebarName').textContent = name;
      document.getElementById('sidebarAvatar').textContent = name.charAt(0).toUpperCase();
      const greeting = document.getElementById('welcomeGreeting');
      if (greeting) {
        greeting.textContent = name !== 'Guest'
          ? `Hello, ${name}! How can I help?`
          : 'Hello! How can I help?';
      }
    }

    // ═══════════════════════════════════════
    //   SIDEBAR
    // ═══════════════════════════════════════
    function toggleSidebar() { sidebar.classList.toggle('collapsed'); }

    function selectChat(el) {
      document.querySelectorAll('.chat-item').forEach(i => i.classList.remove('active'));
      el.classList.add('active');
    }

    // ═══════════════════════════════════════
    //   NEW CHAT
    // ═══════════════════════════════════════
    function startNewChat() {
      history = [];
      lastUserQuestion = '';
      chatInner.innerHTML = `
        <div class="welcome" id="welcome">
          <div class="welcome-icon">
            <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
          </div>
          <h2 id="welcomeGreeting">${currentUser && currentUser !== 'Guest' ? 'Hello, ' + esc(currentUser) + '! How can I help?' : 'Hello! How can I help?'}</h2>
          <p>Ask me anything — I'll search your knowledge base and provide accurate, sourced answers.</p>
          <div class="suggestions">
            <div class="suggestion-card" onclick="sendSuggestion(this)">
              <div class="card-icon"><svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></div>
              <div class="card-title">What is RAG?</div>
              <div class="card-desc">Explain Retrieval-Augmented Generation</div>
            </div>
            <div class="suggestion-card" onclick="sendSuggestion(this)">
              <div class="card-icon"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg></div>
              <div class="card-title">Tell me about NETSOL</div>
              <div class="card-desc">Search company overview and products</div>
            </div>
            <div class="suggestion-card" onclick="sendSuggestion(this)">
              <div class="card-icon"><svg viewBox="0 0 24 24"><path d="M18 20V10M12 20V4M6 20v-6"/></svg></div>
              <div class="card-title">What is a vector database?</div>
              <div class="card-desc">Understand high-dimensional embeddings</div>
            </div>
            <div class="suggestion-card" onclick="sendSuggestion(this)">
              <div class="card-icon"><svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
              <div class="card-title">Explain photosynthesis</div>
              <div class="card-desc">Learn about energy conversion in plants</div>
            </div>
          </div>
        </div>`;
    }

    // ═══════════════════════════════════════
    //   INPUT HELPERS
    // ═══════════════════════════════════════
    function autoResize(el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 140) + 'px';
    }
    function updateSendBtn() {
      sendBtn.disabled = !msgInput.value.trim();
    }
    function handleKey(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }
    function sendSuggestion(card) {
      msgInput.value = card.querySelector('.card-title').textContent;
      sendMessage();
    }
    function getTime() {
      return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    function esc(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }

    // ═══════════════════════════════════════
    //   MARKDOWN RENDERING (simple)
    // ═══════════════════════════════════════
    function renderMd(text) {
      let html = esc(text);
      // Code blocks
      html = html.replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>');
      // Inline code
      html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
      // Bold
      html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
      // Italic
      html = html.replace(/\\*(.+?)\\*/g, '<em>$1</em>');
      // Line breaks
      html = html.replace(/\\n/g, '<br>');
      return html;
    }

    // ═══════════════════════════════════════
    //   MESSAGES
    // ═══════════════════════════════════════
    function createUserMessage(text) {
      const w = document.getElementById('welcome');
      if (w) w.remove();

      const div = document.createElement('div');
      div.className = 'message user';
      const initial = currentUser ? currentUser.charAt(0).toUpperCase() : 'U';
      div.innerHTML = `
        <div class="msg-avatar">${initial}</div>
        <div class="msg-body">
          <div class="msg-bubble">${esc(text)}</div>
          <div class="msg-meta">
            <span class="msg-time">${getTime()}</span>
            <div class="msg-actions">
              <button class="msg-action" title="Copy" onclick="copyMsg(this)">
                <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              </button>
            </div>
          </div>
        </div>
      `;
      chatInner.appendChild(div);
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    function createBotMessage(text, sources) {
      const div = document.createElement('div');
      div.className = 'message bot';

      let html = renderMd(text);

      if (sources && sources.length) {
        const unique = [...new Set(sources.map(s => s.source))];
        html += '<div class="sources">';
        unique.forEach(src => {
          html += `<span class="source-pill">
            <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            ${esc(src)}
          </span>`;
        });
        html += '</div>';
      }

      div.innerHTML = `
        <div class="msg-avatar">
          <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
        </div>
        <div class="msg-body">
          <div class="msg-bubble">${html}</div>
          <div class="msg-meta">
            <span class="msg-time">${getTime()}</span>
            <div class="msg-actions">
              <button class="msg-action tts-btn" title="Read aloud">
                <svg viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
              </button>
              <button class="msg-action" title="Copy" onclick="copyMsg(this)">
                <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              </button>
              <button class="msg-action" title="Regenerate" onclick="regenerateAction()">
                <svg viewBox="0 0 24 24"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
              </button>
              <button class="msg-action" title="Good" onclick="rateFeedback(this,'good')">
                <svg viewBox="0 0 24 24"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
              </button>
              <button class="msg-action" title="Bad" onclick="rateFeedback(this,'bad')">
                <svg viewBox="0 0 24 24"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>
              </button>
            </div>
          </div>
        </div>
      `;

      const ttsBtn = div.querySelector('.tts-btn');
      if (ttsBtn) {
        ttsBtn.dataset.text = text;
        ttsBtn.addEventListener('click', () => playAudio(ttsBtn));
      }
      chatInner.appendChild(div);
      chatArea.scrollTop = chatArea.scrollHeight;
      return div;
    }

    // ═══════════════════════════════════════
    //   TYPING INDICATOR
    // ═══════════════════════════════════════
    function showTyping() {
      const div = document.createElement('div');
      div.className = 'typing';
      div.id = 'typingIndicator';
      div.innerHTML = `
        <div class="msg-avatar" style="background:var(--bg-tertiary);border:1px solid var(--border);color:var(--accent-1);">
          <svg viewBox="0 0 24 24" style="width:16px;height:16px;stroke:var(--accent-1);fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
        </div>
        <div class="typing-dots"><span></span><span></span><span></span></div>
        <span class="typing-label">Searching knowledge base...</span>
      `;
      chatInner.appendChild(div);
      chatArea.scrollTop = chatArea.scrollHeight;
    }
    function removeTyping() {
      const el = document.getElementById('typingIndicator');
      if (el) el.remove();
    }

    // ═══════════════════════════════════════
    //   SEND MESSAGE
    // ═══════════════════════════════════════
    async function sendMessage() {
      const text = msgInput.value.trim();
      if (!text) return;
      lastUserQuestion = text;
      createUserMessage(text);
      msgInput.value = '';
      autoResize(msgInput);
      updateSendBtn();
      showTyping();

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: text, history })
        });
        const data = await res.json();
        removeTyping();
        if (data.answer) {
          createBotMessage(data.answer, data.sources);
          history.push({ role: 'user', content: text });
          history.push({ role: 'assistant', content: data.answer });
        } else {
          createBotMessage(data.detail || 'Something went wrong.', []);
        }
      } catch (err) {
        removeTyping();
        createBotMessage('Connection lost. Is the server still running?', []);
      }
    }

    // ═══════════════════════════════════════
    //   REGENERATE
    // ═══════════════════════════════════════
    async function regenerateAction() {
      if (!lastUserQuestion) return;
      showTyping();
      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: lastUserQuestion, history: history.slice(0, -2) })
        });
        const data = await res.json();
        removeTyping();
        if (data.answer) {
          createBotMessage(data.answer, data.sources);
          history.push({ role: 'user', content: lastUserQuestion });
          history.push({ role: 'assistant', content: data.answer });
        }
      } catch (err) { removeTyping(); }
    }

    // ═══════════════════════════════════════
    //   SCROLL TRACKING
    // ═══════════════════════════════════════
    chatArea.addEventListener('scroll', () => {
      const atBottom = chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < 60;
      scrollBtn.classList.toggle('visible', !atBottom);
    });
    function scrollToBottom() {
      chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
    }

    // ═══════════════════════════════════════
    //   ACTION BUTTONS
    // ═══════════════════════════════════════
    async function playAudio(btn) {
      const text = btn.dataset.text || '';
      try {
        btn.style.color = 'var(--accent-1)';
        const response = await fetch('/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        await audio.play();
        audio.onended = () => { btn.style.color = ''; URL.revokeObjectURL(url); };
      } catch (err) {
        btn.style.color = 'var(--red)';
        setTimeout(() => btn.style.color = '', 2000);
      }
    }

    function copyMsg(btn) {
      const bubble = btn.closest('.msg-body').querySelector('.msg-bubble');
      navigator.clipboard.writeText(bubble.innerText);
      btn.style.color = 'var(--green)';
      setTimeout(() => btn.style.color = '', 1500);
    }

    function rateFeedback(btn, type) {
      btn.style.color = type === 'good' ? 'var(--green)' : 'var(--red)';
      setTimeout(() => btn.style.color = '', 2000);
    }

    // ═══════════════════════════════════════
    //   VOICE INPUT (Web Speech API)
    // ═══════════════════════════════════════
    function toggleMic() {
      if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        showToast('Voice input is not supported in this browser.');
        return;
      }

      if (isRecording) {
        recognition.stop();
        return;
      }

      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognition = new SpeechRecognition();
      recognition.lang = 'en-US';
      recognition.interimResults = true;
      recognition.continuous = false;

      recognition.onstart = () => {
        isRecording = true;
        micBtn.classList.add('recording');
      };
      recognition.onresult = (e) => {
        let transcript = '';
        for (const result of e.results) {
          transcript += result[0].transcript;
        }
        msgInput.value = transcript;
        autoResize(msgInput);
        updateSendBtn();
      };
      recognition.onend = () => {
        isRecording = false;
        micBtn.classList.remove('recording');
      };
      recognition.onerror = (e) => {
        isRecording = false;
        micBtn.classList.remove('recording');
        if (e.error !== 'no-speech') showToast('Voice error: ' + e.error);
      };
      recognition.start();
    }

    // ═══════════════════════════════════════
    //   FILE UPLOAD
    // ═══════════════════════════════════════
    async function handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      event.target.value = ''; // reset

      showToast(`Uploading "${file.name}"...`);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
          showToast(`✓ ${data.message}`);
        } else {
          showToast(`✗ ${data.detail || 'Upload failed'}`, true);
        }
      } catch (err) {
        showToast('✗ Upload error: ' + err.message, true);
      }
    }

    function showToast(msg, isError) {
      const toast = document.getElementById('uploadToast');
      toast.textContent = msg;
      toast.style.borderColor = isError ? 'var(--red)' : 'var(--border)';
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 4000);
    }
  </script>

</body>
</html>
"""