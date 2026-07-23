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
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG-Chatbot</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    *, *::before, *::after {
      margin: 0; padding: 0; box-sizing: border-box;
    }

    :root {
      --bg-primary: #0d0d0f;
      --bg-secondary: #16161a;
      --bg-tertiary: #1e1e24;
      --bg-hover: #2a2a32;
      --bg-active: #32323c;
      --border: #2a2a32;
      --border-light: #3a3a44;
      --text-primary: #f0f0f5;
      --text-secondary: #a0a0ad;
      --text-muted: #666670;
      --accent: #5b7cff;
      --accent-glow: rgba(91, 124, 255, 0.15);
      --accent-hover: #7b98ff;
      --green: #34d399;
      --red: #f87171;
      --user-bubble: #2e2e36;
      --bot-bubble: #18181e;
      --radius: 14px;
      --radius-sm: 10px;
      --radius-xs: 6px;
      --transition: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    html, body {
      height: 100%;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      overflow: hidden;
    }

    .app {
      display: flex;
      height: 100vh;
    }

    /* ═══════════════════════════════════════════
       SIDEBAR
    ═══════════════════════════════════════════ */
    .sidebar {
      width: 280px;
      background: var(--bg-secondary);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      transition: transform var(--transition), opacity var(--transition);
      flex-shrink: 0;
      overflow: hidden;
    }

    .sidebar.collapsed {
      transform: translateX(-100%);
      width: 0;
      opacity: 0;
      pointer-events: none;
      margin-right: 0;
    }

    .sidebar-header {
      padding: 18px 16px 12px;
      border-bottom: 1px solid var(--border);
    }

    .sidebar-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 10px;
    }

    .new-chat-btn {
      width: 100%;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      background: var(--accent);
      border: none;
      border-radius: var(--radius-sm);
      color: #fff;
      font-family: inherit;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: background var(--transition), transform 0.1s;
    }

    .new-chat-btn:hover { background: var(--accent-hover); }
    .new-chat-btn:active { transform: scale(0.98); }

    .new-chat-btn svg {
      width: 16px; height: 16px;
      stroke: currentColor; fill: none;
      stroke-width: 2; stroke-linecap: round;
    }

    .chat-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
    }

    .chat-list::-webkit-scrollbar { width: 4px; }
    .chat-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    .chat-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-radius: var(--radius-xs);
      cursor: pointer;
      transition: background var(--transition);
      margin-bottom: 2px;
    }

    .chat-item:hover { background: var(--bg-hover); }
    .chat-item.active { background: var(--bg-active); }

    .chat-item-icon {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--text-muted);
      flex-shrink: 0;
    }

    .chat-item.active .chat-item-icon { background: var(--accent); }

    .chat-item-text {
      font-size: 13px;
      color: var(--text-secondary);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      flex: 1;
    }

    .chat-item.active .chat-item-text { color: var(--text-primary); }

    .chat-item-time {
      font-size: 10px;
      color: var(--text-muted);
      flex-shrink: 0;
    }

    .sidebar-footer {
      padding: 14px 16px;
      border-top: 1px solid var(--border);
    }

    .user-profile {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      border-radius: var(--radius-xs);
      cursor: pointer;
      transition: background var(--transition);
    }

    .user-profile:hover { background: var(--bg-hover); }

    .user-avatar {
      width: 32px; height: 32px;
      border-radius: 50%;
      background: var(--bg-active);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-secondary);
      border: 1px solid var(--border-light);
    }

    .user-info { flex: 1; }
    .user-name { font-size: 13px; font-weight: 500; color: var(--text-primary); }
    .user-role { font-size: 11px; color: var(--text-muted); }

    /* ═══════════════════════════════════════════
       MAIN AREA
    ═══════════════════════════════════════════ */
    .main {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
      position: relative;
    }

    /* Header */
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 20px;
      border-bottom: 1px solid var(--border);
      background: var(--bg-secondary);
      flex-shrink: 0;
    }

    .header-left {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .toggle-btn {
      width: 34px; height: 34px;
      display: flex; align-items: center; justify-content: center;
      background: transparent; border: none;
      border-radius: var(--radius-xs);
      color: var(--text-secondary);
      cursor: pointer;
      transition: background var(--transition), color var(--transition);
    }

    .toggle-btn:hover { background: var(--bg-hover); color: var(--text-primary); }

    .toggle-btn svg {
      width: 18px; height: 18px;
      stroke: currentColor; fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }

    .header-brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .brand-dot {
      width: 8px; height: 8px;
      background: var(--accent);
      border-radius: 50%;
      animation: glow 2s ease-in-out infinite;
    }

    @keyframes glow {
      0%, 100% { box-shadow: 0 0 0 0 var(--accent-glow); }
      50% { box-shadow: 0 0 8px 3px var(--accent-glow); }
    }

    .brand-name {
      font-size: 14px;
      font-weight: 600;
      letter-spacing: -0.02em;
    }

    .header-right {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .header-btn {
      width: 34px; height: 34px;
      display: flex; align-items: center; justify-content: center;
      background: transparent; border: none;
      border-radius: var(--radius-xs);
      color: var(--text-secondary);
      cursor: pointer;
      transition: background var(--transition), color var(--transition);
      position: relative;
    }

    .header-btn:hover { background: var(--bg-hover); color: var(--text-primary); }

    .header-btn svg {
      width: 16px; height: 16px;
      stroke: currentColor; fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }

    /* ═══════════════════════════════════════════
       CHAT AREA
    ═══════════════════════════════════════════ */
    .chat-area {
      flex: 1;
      overflow-y: auto;
      padding: 28px 32px;
      display: flex;
      flex-direction: column;
      scroll-behavior: smooth;
    }

    .chat-area::-webkit-scrollbar { width: 5px; }
    .chat-area::-webkit-scrollbar-track { background: transparent; }
    .chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

    /* Welcome */
    .welcome {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 56px 24px;
      gap: 14px;
      animation: fadeIn 0.4s ease-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .welcome-logo {
      width: 52px; height: 52px;
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: 16px;
      display: flex; align-items: center; justify-content: center;
      margin-bottom: 4px;
    }

    .welcome-logo svg {
      width: 24px; height: 24px;
      stroke: var(--accent); fill: none;
      stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round;
    }

    .welcome h2 {
      font-size: 20px;
      font-weight: 600;
      letter-spacing: -0.02em;
    }

    .welcome p {
      font-size: 14px;
      color: var(--text-muted);
      line-height: 1.6;
      max-width: 400px;
    }

    .suggestions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
      max-width: 480px;
      width: 100%;
    }

    .suggestion-card {
      padding: 14px 16px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: border-color var(--transition), background var(--transition), transform 0.15s;
      text-align: left;
    }

    .suggestion-card:hover {
      border-color: var(--accent);
      background: var(--bg-tertiary);
      transform: translateY(-1px);
    }

    .suggestion-card:active { transform: translateY(0); }

    .suggestion-card .card-icon {
      width: 28px; height: 28px;
      background: var(--accent-glow);
      border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      margin-bottom: 10px;
    }

    .suggestion-card .card-icon svg {
      width: 14px; height: 14px;
      stroke: var(--accent); fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }

    .suggestion-card .card-title {
      font-size: 13px;
      font-weight: 500;
      color: var(--text-primary);
      margin-bottom: 3px;
    }

    .suggestion-card .card-desc {
      font-size: 11.5px;
      color: var(--text-muted);
      line-height: 1.4;
    }

    /* Messages */
    .message-group {
      display: flex;
      flex-direction: column;
      gap: 16px;
      animation: fadeUp 0.3s ease-out;
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .message {
      display: flex;
      gap: 12px;
      max-width: 78%;
      margin-bottom: 16px;
    }

    .message.user { align-self: flex-end; flex-direction: row-reverse; }
    .message.bot { align-self: flex-start; }

    .avatar {
      width: 32px; height: 32px;
      border-radius: 50%;
      flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 600;
    }

    .message.user .avatar {
      background: var(--accent);
      color: #fff;
    }

    .message.bot .avatar {
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      color: var(--text-secondary);
    }

    .bubble {
      padding: 14px 18px;
      border-radius: var(--radius);
      font-size: 14px;
      line-height: 1.65;
      color: var(--text-primary);
      position: relative;
    }

    .message.user .bubble {
      background: var(--user-bubble);
      border: 1px solid var(--border);
      border-top-right-radius: 4px;
    }

    .message.bot .bubble {
      background: var(--bot-bubble);
      border: 1px solid var(--border);
      border-top-left-radius: 4px;
    }

    .bubble-meta {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 8px;
    }

    .bubble-time {
      font-size: 10.5px;
      color: var(--text-muted);
    }

    .message-actions {
      display: flex;
      gap: 2px;
      opacity: 0;
      transition: opacity var(--transition);
    }

    .message:hover .message-actions { opacity: 1; }

    .action-btn {
      width: 26px; height: 26px;
      display: flex; align-items: center; justify-content: center;
      background: transparent; border: none;
      border-radius: var(--radius-xs);
      color: var(--text-muted);
      cursor: pointer;
      transition: background var(--transition), color var(--transition);
    }

    .action-btn:hover { background: var(--bg-hover); color: var(--text-primary); }

    .action-btn svg {
      width: 13px; height: 13px;
      stroke: currentColor; fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }

    /* Source references */
    .sources {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 12px;
    }

    .source-tag {
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 5px 10px;
      background: var(--bg-primary);
      border: 1px solid var(--border);
      border-radius: var(--radius-xs);
      font-size: 11.5px;
      color: var(--text-secondary);
      cursor: pointer;
      transition: border-color var(--transition), background var(--transition);
    }

    .source-tag:hover {
      border-color: var(--accent);
      background: var(--accent-glow);
    }

    .source-tag svg {
      width: 12px; height: 12px;
      stroke: var(--text-muted); fill: none;
      stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round;
    }

    /* Typing indicator */
    .typing {
      display: flex;
      align-items: center;
      gap: 12px;
      align-self: flex-start;
      margin-bottom: 16px;
      animation: fadeUp 0.2s ease-out;
    }

    .typing-dots {
      display: flex;
      gap: 5px;
      padding: 14px 18px;
      background: var(--bot-bubble);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      border-top-left-radius: 4px;
    }

    .typing-dots span {
      width: 7px; height: 7px;
      background: var(--text-muted);
      border-radius: 50%;
      animation: bounce 1.4s infinite;
    }

    .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.3s; }

    @keyframes bounce {
      0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
      40% { opacity: 1; transform: translateY(-4px); }
    }

    /* ═══════════════════════════════════════════
       INPUT AREA
    ═══════════════════════════════════════════ */
    .input-area {
      flex-shrink: 0;
      padding: 0 32px 16px;
    }

    .input-wrapper {
      display: flex;
      align-items: flex-end;
      gap: 8px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 8px 10px;
      transition: border-color var(--transition), box-shadow var(--transition);
    }

    .input-wrapper:focus-within {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }

    .input-wrapper textarea {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      resize: none;
      padding: 10px 12px;
      font-family: inherit;
      font-size: 14px;
      color: var(--text-primary);
      line-height: 1.5;
      max-height: 130px;
      min-height: 22px;
    }

    .input-wrapper textarea::placeholder { color: var(--text-muted); }

    .input-actions {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .input-action-btn {
      width: 34px; height: 34px;
      display: flex; align-items: center; justify-content: center;
      background: transparent; border: none;
      border-radius: var(--radius-xs);
      color: var(--text-muted);
      cursor: pointer;
      transition: background var(--transition), color var(--transition);
    }

    .input-action-btn:hover { background: var(--bg-hover); color: var(--text-secondary); }

    .input-action-btn svg {
      width: 16px; height: 16px;
      stroke: currentColor; fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }

    .send-btn {
      width: 36px; height: 36px;
      display: flex; align-items: center; justify-content: center;
      background: var(--accent);
      border: none;
      border-radius: var(--radius-xs);
      cursor: pointer;
      transition: background var(--transition), transform 0.1s, opacity var(--transition);
      flex-shrink: 0;
    }

    .send-btn:hover { background: var(--accent-hover); }
    .send-btn:active { transform: scale(0.92); }
    .send-btn:disabled { opacity: 0.4; cursor: default; }

    .send-btn svg {
      width: 16px; height: 16px;
      stroke: #fff; fill: none;
      stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;
    }

    .input-footer {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding-top: 10px;
    }

    .input-footer span {
      font-size: 11px;
      color: var(--text-muted);
    }

    .input-footer .dot {
      width: 3px; height: 3px;
      background: var(--text-muted);
      border-radius: 50%;
    }

    /* Scroll-to-bottom button */
    .scroll-btn {
      position: absolute;
      bottom: 90px;
      right: 48px;
      width: 36px; height: 36px;
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      color: var(--text-secondary);
      cursor: pointer;
      opacity: 0;
      transform: translateY(8px);
      transition: opacity var(--transition), transform var(--transition), background var(--transition);
      pointer-events: none;
      z-index: 10;
    }

    .scroll-btn.visible {
      opacity: 1;
      transform: translateY(0);
      pointer-events: all;
    }

    .scroll-btn:hover { background: var(--bg-hover); color: var(--text-primary); }

    .scroll-btn svg {
      width: 16px; height: 16px;
      stroke: currentColor; fill: none;
      stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;
    }

    /* Context menu */
    .context-menu {
      position: fixed;
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 6px;
      z-index: 200;
      min-width: 160px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      display: none;
    }

    .context-menu.show { display: block; animation: fadeUp 0.15s ease-out; }

    .context-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: var(--radius-xs);
      font-size: 12.5px;
      color: var(--text-secondary);
      cursor: pointer;
      transition: background var(--transition), color var(--transition);
    }

    .context-item:hover { background: var(--bg-hover); color: var(--text-primary); }

    .context-item svg {
      width: 14px; height: 14px;
      stroke: currentColor; fill: none;
      stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
    }

    .context-divider {
      height: 1px;
      background: var(--border);
      margin: 4px 6px;
    }

    @media (max-width: 768px) {
      .sidebar { position: absolute; height: 100%; z-index: 50; }
      .chat-area { padding: 20px 16px; }
      .input-area { padding: 0 16px 16px; }
      .suggestions { grid-template-columns: 1fr; }
      .message { max-width: 90%; }
    }
  </style>
</head>
<body>

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
        <div class="user-profile">
          <div class="user-avatar">U</div>
          <div class="user-info">
            <div class="user-name">User</div>
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
          <button class="toggle-btn" onclick="toggleSidebar()" title="Toggle sidebar">
            <svg viewBox="0 0 24 24"><path d="M3 12h18M3 6h18M3 18h18"/></svg>
          </button>
          <div class="header-brand">
            <div class="brand-dot"></div>
            <span class="brand-name">RAG-Chatbot</span>
          </div>
        </div>
        <div class="header-right">
          <button class="header-btn" title="New Chat" onclick="startNewChat()">
            <svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>
          </button>
          <button class="header-btn" title="Settings" onclick="toggleSettings()">
            <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          </button>
        </div>
      </div>

      <!-- Chat Area -->
      <div class="chat-area" id="chatArea">

        <div class="welcome" id="welcome">
          <div class="welcome-logo">
            <svg viewBox="0 0 24 24"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
          </div>
          <h2>How can I help you today?</h2>
          <p>Ask me anything. I'll search your knowledge base and provide accurate, sourced answers.</p>
          <div class="suggestions">
            <div class="suggestion-card" onclick="sendSuggestion(this)">
              <div class="card-icon">
                <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
              </div>
              <div class="card-title">What is RAG?</div>
              <div class="card-desc">Explain Retrieval-Augmented Generation</div>
            </div>
            <div class="suggestion-card" onclick="sendSuggestion(this)">
              <div class="card-icon">
                <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
              </div>
              <div class="card-title">Tell me about NETSOL</div>
              <div class="card-desc">Search company overview and products</div>
            </div>
            <div class="suggestion-card" onclick="sendSuggestion(this)">
              <div class="card-icon">
                <svg viewBox="0 0 24 24"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>
              </div>
              <div class="card-title">What is a vector database?</div>
              <div class="card-desc">Understand high-dimensional embeddings</div>
            </div>
            <div class="suggestion-card" onclick="sendSuggestion(this)">
              <div class="card-icon">
                <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
              </div>
              <div class="card-title">Explain photosynthesis</div>
              <div class="card-desc">Learn about energy conversion in plants</div>
            </div>
          </div>
        </div>

      </div>

      <!-- Scroll to bottom -->
      <button class="scroll-btn" id="scrollBtn" onclick="scrollToBottom()">
        <svg viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>
      </button>

      <!-- Input -->
      <div class="input-area">
        <div class="input-wrapper" id="inputWrapper">
          <button class="input-action-btn" title="Attach file (coming soon)">
            <svg viewBox="0 0 24 24"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
          </button>
          <textarea
            id="msgInput"
            rows="1"
            placeholder="Ask anything..."
            oninput="autoResize(this); updateSendBtn()"
            onkeydown="handleKey(event)"
          ></textarea>
          <div class="input-actions">
            <button class="send-btn" id="sendBtn" onclick="sendMessage()" disabled title="Send message">
              <svg viewBox="0 0 24 24"><path d="M22 2L11 13"/><path d="M22 2L15 22l-4-9-9-4z"/></svg>
            </button>
          </div>
        </div>
        <div class="input-footer">
          <span>RAG-Chatbot</span>
          <div class="dot"></div>
          <span>Grounded in your data</span>
          <div class="dot"></div>
          <span>Press Enter to send</span>
        </div>
      </div>

    </div>
  </div>

  <!-- Context Menu -->
  <div class="context-menu" id="contextMenu">
    <div class="context-item" onclick="copyContextMenuText()">
      <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      Copy text
    </div>
    <div class="context-item" onclick="regenerateLastResponse()">
      <svg viewBox="0 0 24 24"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
      Regenerate
    </div>
    <div class="context-divider"></div>
    <div class="context-item" onclick="editContextMessage()">
      <svg viewBox="0 0 24 24"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
      Edit message
    </div>
  </div>

  <script>
    const chatArea = document.getElementById('chatArea');
    const msgInput = document.getElementById('msgInput');
    const sendBtn = document.getElementById('sendBtn');
    const scrollBtn = document.getElementById('scrollBtn');
    const contextMenu = document.getElementById('contextMenu');
    const sidebar = document.getElementById('sidebar');

    let history = [];
    let lastUserQuestion = '';
    let selectedMessageEl = null;

    // ── Sidebar toggle ──
    function toggleSidebar() {
      sidebar.classList.toggle('collapsed');
    }

    function selectChat(el) {
      document.querySelectorAll('.chat-item').forEach(i => i.classList.remove('active'));
      el.classList.add('active');
    }

    // ── New chat ──
    function startNewChat() {
      history = [];
      lastUserQuestion = '';
      chatArea.innerHTML = `
        <div class="welcome" id="welcome">
          <div class="welcome-logo">
            <svg viewBox="0 0 24 24"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
          </div>
          <h2>How can I help you today?</h2>
          <p>Ask me anything. I'll search your knowledge base and provide accurate, sourced answers.</p>
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

    // ── Auto-resize textarea ──
    function autoResize(el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 130) + 'px';
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

    function escapeHtml(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }

    // ── Create User Message ──
    function createUserMessage(text) {
      const w = document.getElementById('welcome');
      if (w) w.remove();

      const div = document.createElement('div');
      div.className = 'message user';
      div.oncontextmenu = function(e) { e.preventDefault(); showContext(e, this); };

      div.innerHTML = `
        <div class="avatar">U</div>
        <div class="bubble">
          ${escapeHtml(text)}
          <div class="bubble-meta">
            <span class="bubble-time">${getTime()}</span>
            <div class="message-actions">
              <button class="action-btn" title="Copy" onclick="copyText(this)">
                <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              </button>
            </div>
          </div>
        </div>
      `;
      chatArea.appendChild(div);
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    // ── Create Bot Response ──
    function createBotMessage(text, sources) {
      const div = document.createElement('div');
      div.className = 'message bot';
      div.oncontextmenu = function(e) { e.preventDefault(); showContext(e, this); };

      let html = escapeHtml(text).replace(/\\n/g, '<br>');

      if (sources && sources.length) {
        const unique = [...new Set(sources.map(s => s.source))];
        html += '<div class="sources">';
        unique.forEach(src => {
          html += `<span class="source-tag">
            <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            ${escapeHtml(src)}
          </span>`;
        });
        html += '</div>';
      }

      div.innerHTML = `
        <div class="avatar" style="background:var(--bg-tertiary);border:1px solid var(--border);color:var(--text-secondary);">R</div>
        <div class="bubble">
          ${html}
          <div class="bubble-meta">
            <span class="bubble-time">${getTime()}</span>
            <div class="message-actions">
              <button class="action-btn" title="Copy" onclick="copyText(this)">
                <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              </button>
              <button class="action-btn" title="Regenerate" onclick="regenerateAction(this)">
                <svg viewBox="0 0 24 24"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
              </button>
              <button class="action-btn" title="Good response" onclick="rateGood(this)">
                <svg viewBox="0 0 24 24"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
              </button>
              <button class="action-btn" title="Bad response" onclick="rateBad(this)">
                <svg viewBox="0 0 24 24"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>
              </button>
            </div>
          </div>
        </div>
      `;
      chatArea.appendChild(div);
      chatArea.scrollTop = chatArea.scrollHeight;
      return div;
    }

    // ── Typing Indicator ──
    function showTyping() {
      const div = document.createElement('div');
      div.className = 'typing';
      div.id = 'typingIndicator';
      div.innerHTML = `
        <div class="avatar" style="background:var(--bg-tertiary);border:1px solid var(--border);color:var(--text-secondary);">R</div>
        <div class="typing-dots"><span></span><span></span><span></span></div>
        <span style="font-size:12px;color:var(--text-muted);margin-left:4px;">Searching knowledge base...</span>
      `;
      chatArea.appendChild(div);
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    function removeTyping() {
      const el = document.getElementById('typingIndicator');
      if (el) el.remove();
    }

    // ── Send Message ──
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
          body: JSON.stringify({ question: text, history: history })
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

    // ── Regenerate Action ──
    async function regenerateAction(btn) {
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
      } catch (err) {
        removeTyping();
      }
    }

    // ── Scroll tracking ──
    chatArea.addEventListener('scroll', () => {
      const atBottom = chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < 60;
      scrollBtn.classList.toggle('visible', !atBottom);
    });

    function scrollToBottom() {
      chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
    }

    // ── Action Buttons ──
    function copyText(btn) {
      const bubble = btn.closest('.bubble');
      const textNode = bubble.firstChild;
      const text = textNode ? textNode.textContent.trim() : bubble.innerText;
      navigator.clipboard.writeText(text);
      btn.style.color = 'var(--green)';
      setTimeout(() => btn.style.color = '', 1500);
    }

    function rateGood(btn) {
      btn.style.color = 'var(--green)';
      setTimeout(() => btn.style.color = '', 2000);
    }

    function rateBad(btn) {
      btn.style.color = 'var(--red)';
      setTimeout(() => btn.style.color = '', 2000);
    }

    // ── Context Menu ──
    function showContext(e, el) {
      selectedMessageEl = el;
      contextMenu.style.left = e.clientX + 'px';
      contextMenu.style.top = e.clientY + 'px';
      contextMenu.classList.add('show');
    }

    document.addEventListener('click', () => contextMenu.classList.remove('show'));

    function copyContextMenuText() {
      if (selectedMessageEl) {
        const bubble = selectedMessageEl.querySelector('.bubble');
        if (bubble) {
          const text = bubble.firstChild ? bubble.firstChild.textContent.trim() : bubble.innerText;
          navigator.clipboard.writeText(text);
        }
      }
    }

    function regenerateLastResponse() {
      if (lastUserQuestion) {
        regenerateAction(null);
      }
    }

    function editContextMessage() {
      if (selectedMessageEl) {
        const bubble = selectedMessageEl.querySelector('.bubble');
        if (bubble) {
          const text = bubble.firstChild ? bubble.firstChild.textContent.trim() : bubble.innerText;
          msgInput.value = text;
          autoResize(msgInput);
          updateSendBtn();
        }
      }
    }

    function toggleSettings() {
      alert("Settings modal: Connected to RAG Qdrant Cloud & Groq Llama-3.1 API.");
    }
  </script>

</body>
</html>
"""