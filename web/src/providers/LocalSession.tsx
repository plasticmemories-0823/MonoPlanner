"use client";
import { useEffect, useState, type ReactNode } from "react";

export function LocalSession({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    for (const key of ["lg:chat:apiKey", "lg:chat:apiKeyHost"])
      try { localStorage.removeItem(key); } catch { /* Browser storage may be unavailable. */ }
    // Retire obsolete connection controls without retaining their values.
    const url = new URL(location.href);
    for (const key of ["apiUrl", "assistantId", "authScheme"]) url.searchParams.delete(key);
    history.replaceState(history.state, "", url);
    fetch("/session", { method: "POST", signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw new Error("无法连接本地服务，请检查启动器是否正常运行。");
      setReady(true);
      setError("");
    }).catch((cause) => { if (!controller.signal.aborted) setError(cause.message); });
    return () => controller.abort();
  }, [attempt]);
  if (!ready) return <div className="flex min-h-screen flex-col items-center justify-center gap-4 p-6">
    <p>{error || "正在连接 MonoPlanner…"}</p>
    {error && <button className="rounded-lg border px-4 py-2" onClick={() => setAttempt(attempt + 1)}>重新连接</button>}
  </div>;
  return children;
}
