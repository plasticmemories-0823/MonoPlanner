import "server-only";
import { createHmac, timingSafeEqual } from "node:crypto";

export const SESSION_COOKIE = "monoplanner-session";
export const MAX_BODY = 8 * 1024 * 1024;

export function instanceToken(): string {
  const value = process.env.MONOPLANER_SERVICE_TOKEN || "";
  if (value.length < 32) throw new Error("Local service access is not initialized");
  return value;
}

export function sessionValue(): string {
  return createHmac("sha256", instanceToken()).update("MonoPlanner browser session v1").digest("hex");
}

export function validSession(value: string | undefined): boolean {
  const expected = Buffer.from(sessionValue());
  const actual = Buffer.from(value || "");
  return actual.length === expected.length && timingSafeEqual(actual, expected);
}

export function allowedRequest(request: Request, requireOrigin = false): boolean {
  const host = request.headers.get("host") || "";
  const port = process.env.MONOPLANER_UI_PORT || "3000";
  const allowed = ["127.0.0.1", "localhost", "monoplanner.localhost"].map((name) => `${name}:${port}`);
  if (!allowed.includes(host.toLowerCase())) return false;
  const origin = request.headers.get("origin");
  if ((requireOrigin && !origin) || (origin && origin !== `http://${host}`)) return false;
  const site = request.headers.get("sec-fetch-site");
  return !site || site === "same-origin" || site === "none";
}

export function backendUrl(): string {
  // Only the launcher sets this port; no browser query, header or persisted option can replace it.
  const port = Number(process.env.MONOPLANER_BACKEND_PORT || "2024");
  if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error("Invalid local service port");
  return `http://127.0.0.1:${port}`;
}
