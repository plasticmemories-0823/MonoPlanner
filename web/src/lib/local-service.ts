/** The browser never selects a backend host or retains service credentials. */
export function localApiUrl(): string {
  return typeof window === "undefined" ? "/api" : `${window.location.origin}/api`;
}
