// Thin client for the FastAPI backend. The JWT lives in localStorage (see README: a
// production build would use an httpOnly cookie set by the backend instead).

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "cardocs.token";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // storage unavailable (private mode); the session just won't persist
  }
}

function detailOf(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg);
  }
  return fallback;
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData) && !(init.body instanceof URLSearchParams)) {
    headers.set("Content-Type", "application/json");
  }

  let resp: Response;
  try {
    resp = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, `Can't reach the server at ${API_URL}. Is the API running?`);
  }

  if (resp.status === 401 && path !== "/auth/login") {
    setToken(null);
    // Outside React here, so a full navigation is the simplest way back to the login page.
    // eslint-disable-next-line @next/next/no-location-assign-relative-destination
    window.location.href = "/login";
    throw new ApiError(401, "Your session ended. Sign in again.");
  }
  if (resp.status === 204) return undefined as T;

  const body = await resp.json().catch(() => null);
  if (!resp.ok) throw new ApiError(resp.status, detailOf(body, `Request failed (${resp.status})`));
  return body as T;
}

export async function login(email: string, password: string): Promise<void> {
  const form = new URLSearchParams({ username: email, password });
  const { access_token } = await api<{ access_token: string }>("/auth/login", {
    method: "POST",
    body: form,
  });
  setToken(access_token);
}

/** Open a stored PDF in a new tab at the given page. */
export async function openDocumentPage(documentId: number, page: number): Promise<void> {
  // Open the tab synchronously so popup blockers allow it, then point it at the signed URL.
  const tab = window.open("", "_blank");
  try {
    const { url } = await api<{ url: string }>(`/documents/${documentId}/file`);
    if (tab) tab.location.href = `${url}#page=${page}`;
    // External signed URL, not an app route.
    // eslint-disable-next-line @next/next/no-location-assign-relative-destination
    else window.location.href = `${url}#page=${page}`;
  } catch (err) {
    tab?.close();
    throw err;
  }
}
