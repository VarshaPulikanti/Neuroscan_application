export interface PredictionResponse {
  prediction: string;
  prediction_label: string;
  confidence: number;
  probabilities: Record<string, number>;
  gradcam_image?: string | null;
  scan_id?: number | null;
}

export interface User {
  id: number;
  email: string;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface ScanSummary {
  id: number;
  filename: string;
  prediction_label: string;
  confidence: number;
  created_at: string;
}

export interface ScanDetail extends ScanSummary {
  prediction: string;
  probabilities: Record<string, number>;
  image_thumbnail?: string | null;
  gradcam_image?: string | null;
}

const API_BASE =
  import.meta.env.VITE_API_URL?.replace(/\/$/, "") ??
  (import.meta.env.DEV ? "/api" : "");

const TOKEN_KEY = "neuroscan_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(): HeadersInit {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(
      typeof err.detail === "string" ? err.detail : "Request failed"
    );
  }
  return res.json();
}

export async function register(
  email: string,
  password: string
): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return handleResponse<AuthResponse>(res);
}

export async function login(
  email: string,
  password: string
): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return handleResponse<AuthResponse>(res);
}

export async function fetchMe(): Promise<User> {
  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: authHeaders(),
  });
  return handleResponse<User>(res);
}

export async function analyzeImage(
  file: File,
  gradcam: boolean
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/predict?gradcam=${gradcam}`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });

  return handleResponse<PredictionResponse>(res);
}

export async function listScans(): Promise<ScanSummary[]> {
  const res = await fetch(`${API_BASE}/scans`, { headers: authHeaders() });
  return handleResponse<ScanSummary[]>(res);
}

export async function getScan(id: number): Promise<ScanDetail> {
  const res = await fetch(`${API_BASE}/scans/${id}`, { headers: authHeaders() });
  return handleResponse<ScanDetail>(res);
}

export async function deleteScan(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/scans/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Delete failed" }));
    throw new Error(err.detail || "Delete failed");
  }
}
