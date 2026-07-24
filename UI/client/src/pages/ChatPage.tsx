/*
 * Ethereal Glass Design — Chat Page
 * Wired to the RAG-Chatbot FastAPI backend (/chat, /upload).
 */
import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BotAvatar } from "@/components/BotAvatar";
import { ParticleBackground } from "@/components/ParticleBackground";
import { SuggestedPrompts } from "@/components/SuggestedPrompts";
import { useAuth } from "@/contexts/AuthContext";
import { useLocation } from "wouter";
import { toast } from "sonner";
import { Plus, Send, Mic, Paperclip, Menu, X, LogOut } from "lucide-react";
import { Streamdown } from "streamdown";
import { chat, uploadDocument, type Source } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: Source[];
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  timestamp: Date;
}

export default function ChatPage() {
  const { user, logout, isAuthenticated } = useAuth();
  const [, navigate] = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [activeConversation, setActiveConversation] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [conversations, setConversations] = useState<Conversation[]>([
    { id: "1", title: "New Conversation", messages: [], timestamp: new Date() },
  ]);

  // Redirect to auth if not signed in
  useEffect(() => {
    if (!isAuthenticated) navigate("/auth");
  }, [isAuthenticated, navigate]);

  const currentConversation = activeConversation
    ? conversations.find((c) => c.id === activeConversation)
    : conversations[0];

  const currentMessages = currentConversation?.messages || [];
  const hasMessages = currentMessages.length > 0;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentMessages, isTyping]);

  const autoResize = useCallback(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 128) + "px";
    }
  }, []);

  useEffect(() => {
    autoResize();
  }, [inputValue, autoResize]);

  const sendMessage = useCallback(
    async (content: string) => {
      const text = content.trim();
      if (!text || isTyping) return;

      const convId = activeConversation || conversations[0]?.id;
      const conv = conversations.find((c) => c.id === convId);
      const priorHistory = (conv?.messages || []).map((m) => ({ role: m.role, content: m.content }));

      const userMsg: Message = {
        id: `msg-${Date.now()}`,
        role: "user",
        content: text,
        timestamp: new Date(),
      };

      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: [...c.messages, userMsg],
                title: c.messages.length === 0 ? text.slice(0, 40) : c.title,
              }
            : c,
        ),
      );
      setInputValue("");
      setIsTyping(true);

      try {
        const res = await chat(text, priorHistory);
        const botMsg: Message = {
          id: `msg-${Date.now()}-bot`,
          role: "assistant",
          content: res.answer,
          timestamp: new Date(),
          sources: res.sources,
        };
        setConversations((prev) =>
          prev.map((c) => (c.id === convId ? { ...c, messages: [...c.messages, botMsg] } : c)),
        );
      } catch (err: any) {
        const detail = err?.response?.data?.detail || err?.message || "Something went wrong.";
        const errMsg: Message = {
          id: `msg-${Date.now()}-err`,
          role: "assistant",
          content: `⚠️ ${detail}`,
          timestamp: new Date(),
        };
        setConversations((prev) =>
          prev.map((c) => (c.id === convId ? { ...c, messages: [...c.messages, errMsg] } : c)),
        );
      } finally {
        setIsTyping(false);
      }
    },
    [isTyping, activeConversation, conversations],
  );

  const handleSend = () => sendMessage(inputValue);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    const newConv: Conversation = {
      id: `conv-${Date.now()}`,
      title: "New Conversation",
      messages: [],
      timestamp: new Date(),
    };
    setConversations((prev) => [newConv, ...prev]);
    setActiveConversation(newConv.id);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    toast.loading(`Uploading "${file.name}"…`, { id: "upload" });
    try {
      const res = await uploadDocument(file);
      toast.success(res.message || `Ingested ${file.name}`, { id: "upload" });
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || "Upload failed", { id: "upload" });
    }
  };

  const handlePromptSelect = (prompt: string) => sendMessage(prompt);

  return (
    <div className="min-h-screen bg-background relative overflow-hidden flex">
      {/* Background */}
      <div className="fixed inset-0 bg-gradient-to-br from-background via-background/95 to-background/90" />
      <ParticleBackground count={10} />

      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ x: -280, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -280, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
            className="relative z-20 w-72 h-screen flex flex-col"
            style={{
              background: "oklch(0.10 0.02 270 / 0.6)",
              backdropFilter: "blur(24px)",
              borderRight: "1px solid oklch(1 0 0 / 0.06)",
            }}
          >
            <div className="p-4 border-b border-white/[0.06]">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-2 h-2 rounded-full bg-teal shadow-[0_0_6px_oklch(0.82_0.15_175_/0.5)]" />
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Conversations
                </span>
              </div>
              <button
                onClick={handleNewChat}
                className="w-full h-10 rounded-xl text-sm font-medium flex items-center justify-center gap-2 transition-all duration-200 active:scale-[0.97]"
                style={{
                  background:
                    "linear-gradient(135deg, oklch(0.82 0.15 175 / 0.15), oklch(0.62 0.15 275 / 0.1))",
                  border: "1px solid oklch(0.82 0.15 175 / 0.2)",
                  color: "var(--color-teal)",
                }}
              >
                <Plus className="w-4 h-4" />
                New Chat
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-1">
              {conversations.map((conv, i) => (
                <motion.button
                  key={conv.id}
                  onClick={() => setActiveConversation(conv.id)}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05, duration: 0.3 }}
                  className={`w-full text-left px-3 py-2.5 rounded-xl text-sm transition-all duration-200 ${
                    conv.id === (activeConversation || conversations[0]?.id)
                      ? "bg-teal/10 border border-teal/20 text-foreground"
                      : "text-muted-foreground hover:bg-white/[0.04] hover:text-foreground/80 border border-transparent"
                  }`}
                  style={{ fontFamily: "var(--font-sans)" }}
                >
                  <div className="flex items-center justify-between">
                    <span className="truncate">{conv.title}</span>
                    <span className="text-[10px] text-muted-foreground/50 ml-2">
                      {conv.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </span>
                  </div>
                </motion.button>
              ))}
            </div>

            <div className="p-4 border-t border-white/[0.06]">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-full bg-gradient-to-br from-teal to-violet flex items-center justify-center text-xs font-bold text-white">
                  {user?.name?.[0]?.toUpperCase() || "U"}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{user?.name || "User"}</p>
                  <p className="text-xs text-muted-foreground">RAG Workspace</p>
                </div>
                <button
                  onClick={() => {
                    logout();
                    navigate("/auth");
                  }}
                  className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/[0.04] transition-all"
                  title="Sign out"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="relative z-10 flex-1 flex flex-col h-screen">
        <header
          className="relative h-14 flex items-center justify-between px-4 border-b border-white/[0.06]"
          style={{ background: "oklch(0.12 0.02 270 / 0.5)", backdropFilter: "blur(20px)" }}
        >
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/[0.04] transition-all"
            >
              {sidebarOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </button>
            <div className="flex items-center gap-2">
              <motion.div
                className="w-2 h-2 rounded-full bg-teal"
                animate={{
                  boxShadow: [
                    "0 0 0px oklch(0.82 0.15 175 / 0.3)",
                    "0 0 8px oklch(0.82 0.15 175 / 0.5)",
                    "0 0 0px oklch(0.82 0.15 175 / 0.3)",
                  ],
                }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <h1 className="text-sm font-semibold text-foreground" style={{ fontFamily: "var(--font-display)" }}>
                RAG-Chatbot
              </h1>
            </div>
          </div>
          <BotAvatar size="sm" />
        </header>

        {/* Messages / Welcome */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-6 min-h-full flex flex-col justify-end">
            {!hasMessages && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.6 }}
                className="flex flex-col items-center justify-center py-16 space-y-8"
              >
                <motion.div
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.2, duration: 0.5, ease: [0.34, 1.56, 0.64, 1] }}
                >
                  <BotAvatar size="lg" showGreeting={true} />
                </motion.div>
                <div className="text-center space-y-2">
                  <h2
                    className="text-2xl font-bold text-glow-teal"
                    style={{ fontFamily: "var(--font-display)", color: "var(--color-teal)" }}
                  >
                    Ask anything. Know everything.
                  </h2>
                  <p className="text-sm text-muted-foreground max-w-md">
                    {user?.name && user.name !== "Guest" ? `Welcome, ${user.name}. ` : ""}
                    Your AI companion searches your knowledge base and answers with sources.
                  </p>
                </div>
                <SuggestedPrompts onSelect={handlePromptSelect} />
              </motion.div>
            )}

            <AnimatePresence>
              {currentMessages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 12, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.4, ease: [0.34, 1.56, 0.64, 1] as any }}
                  className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
                >
                  {msg.role === "assistant" ? (
                    <BotAvatar size="sm" className="flex-shrink-0 mt-1" />
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-teal to-violet flex items-center justify-center flex-shrink-0 text-xs font-bold text-white mt-1">
                      {user?.name?.[0]?.toUpperCase() || "U"}
                    </div>
                  )}

                  <div className={`max-w-[80%] ${msg.role === "user" ? "items-end" : "items-start"}`}>
                    <div
                      className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                        msg.role === "user"
                          ? "bg-teal/10 border border-teal/20 text-foreground rounded-tr-md"
                          : "glass text-foreground/90 rounded-tl-md"
                      }`}
                      style={{ backdropFilter: msg.role === "assistant" ? "blur(12px)" : undefined }}
                    >
                      {msg.role === "assistant" ? (
                        <Streamdown>{msg.content}</Streamdown>
                      ) : (
                        <p>{msg.content}</p>
                      )}
                    </div>

                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {[...new Set(msg.sources.map((s) => s.source))].map((src) => (
                          <div
                            key={src}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal/10 border border-teal/20 w-fit"
                          >
                            <Paperclip className="w-3 h-3 text-teal" />
                            <span className="text-xs text-teal">{src}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    <p className="text-[10px] text-muted-foreground/50 mt-1.5 px-1">
                      {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </p>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            <AnimatePresence>
              {isTyping && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 5 }}
                  className="flex gap-3"
                >
                  <BotAvatar size="sm" className="flex-shrink-0" />
                  <div
                    className="glass rounded-2xl rounded-tl-md px-4 py-3 flex items-center gap-1.5"
                    style={{ backdropFilter: "blur(12px)" }}
                  >
                    {[0, 1, 2].map((i) => (
                      <motion.div
                        key={i}
                        className="w-2 h-2 rounded-full"
                        style={{ background: "oklch(0.82 0.15 175 / 0.6)" }}
                        animate={{ y: [0, -5, 0], opacity: [0.4, 1, 0.4] }}
                        transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.15 }}
                      />
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div ref={messagesEndRef} className="h-1" />
          </div>
        </div>

        {/* Input Area */}
        <div className="relative z-20 px-4 pb-4 pt-2">
          <div className="max-w-3xl mx-auto">
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,.md,.pdf"
              onChange={handleFileUpload}
              className="hidden"
            />
            <motion.div
              className="rounded-2xl border border-white/[0.08] p-1.5 flex items-end gap-1.5"
              style={{ background: "oklch(0.13 0.02 270 / 0.7)", backdropFilter: "blur(20px)" }}
            >
              <button
                onClick={() => fileInputRef.current?.click()}
                title="Upload a document (.txt, .md, .pdf)"
                className="p-2 rounded-xl text-muted-foreground hover:text-teal hover:bg-teal/10 transition-all flex-shrink-0"
              >
                <Plus className="w-4 h-4" />
              </button>

              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything..."
                rows={1}
                className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground/40 resize-none outline-none py-2.5 px-2 max-h-32"
                style={{ fontFamily: "var(--font-sans)" }}
              />

              <div className="flex items-center gap-1 flex-shrink-0">
                <button
                  className="p-2 rounded-xl text-muted-foreground/30 cursor-not-allowed"
                  disabled
                  title="Voice input coming soon"
                >
                  <Mic className="w-4 h-4" />
                </button>
                <motion.button
                  onClick={handleSend}
                  disabled={!inputValue.trim() || isTyping}
                  className="h-9 w-9 rounded-xl flex items-center justify-center transition-all active:scale-[0.93]"
                  animate={{
                    background:
                      inputValue.trim() && !isTyping
                        ? "linear-gradient(135deg, oklch(0.82 0.15 175), oklch(0.72 0.14 180))"
                        : "oklch(0.2 0.02 270)",
                    boxShadow:
                      inputValue.trim() && !isTyping ? "0 0 16px oklch(0.82 0.15 175 / 0.35)" : "none",
                  }}
                  whileTap={{ scale: 0.93 }}
                >
                  <Send className="w-4 h-4 text-white" />
                </motion.button>
              </div>
            </motion.div>

            <div className="flex items-center justify-center gap-3 mt-3 text-[10px] text-muted-foreground/40">
              <span>RAG-Chatbot</span>
              <span>•</span>
              <span>Grounded in your data</span>
              <span>•</span>
              <span>Enter to send</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
