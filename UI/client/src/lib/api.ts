/*
 * API client for the RAG-Chatbot FastAPI backend.
 * In dev the frontend runs on :3000 and the backend on :8000 (CORS is open on
 * the backend). In production set VITE_API_URL to the backend origin, or leave
 * it empty to call the same origin the app is served from.
 */
import axios from "axios";

export const API_BASE =
  (import.meta.env.VITE_API_URL as string | undefined) ??
  (import.meta.env.DEV ? "http://localhost:8000" : "");

export const api = axios.create({ baseURL: API_BASE });

export interface Source {
  content?: string;
  source: string;
}

export interface ChatResult {
  answer: string;
  sources: Source[];
}

export interface AuthResult {
  authenticated: boolean;
  username: string;
  message: string;
}

export interface UploadResult {
  status: string;
  filename: string;
  chunks: number;
  message: string;
}

export async function chat(
  question: string,
  history: { role: string; content: string }[],
): Promise<ChatResult> {
  const { data } = await api.post("/chat", { question, history });
  return data as ChatResult;
}

export async function registerFace(username: string, image: string): Promise<AuthResult> {
  const { data } = await api.post("/auth/register", { username, image });
  return data as AuthResult;
}

export async function loginFace(image: string): Promise<AuthResult> {
  const { data } = await api.post("/auth/login", { image });
  return data as AuthResult;
}

export async function uploadDocument(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/upload", form);
  return data as UploadResult;
}

/** Fetch TTS audio for `text` and return an object URL for an <audio> element. */
export async function ttsAudioUrl(text: string): Promise<string> {
  const { data } = await api.post("/tts", { text }, { responseType: "blob" });
  return URL.createObjectURL(data as Blob);
}
