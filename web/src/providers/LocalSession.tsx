"use client";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { connectLocalSession } from "@/lib/local-session";
import { readStorageFailure } from "@/lib/conversation-api";
import type { StorageFailure } from "@/lib/conversation-types";

const ServiceHealth = createContext<StorageFailure | null>(null);
export const useInitialServiceHealth = () => useContext(ServiceHealth);

export function LocalSession({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [storageError, setStorageError] = useState<StorageFailure | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    for (const key of ["lg:chat:apiKey", "lg:chat:apiKeyHost"])
      try {
        localStorage.removeItem(key);
      } catch {
        /* 浏览器存储可能不可用。 */
      }
    // 移除过时的连接设置，不保留其值。
    const url = new URL(location.href);
    for (const key of ["apiUrl", "assistantId", "authScheme"]) url.searchParams.delete(key);
    history.replaceState(history.state, "", url);
    connectLocalSession(controller.signal)
      .then((service) => {
        setStorageError(readStorageFailure(service.storage_error));
        setReady(true);
        setError("");
      })
      .catch((cause) => {
        if (!controller.signal.aborted) setError(cause.message);
      });
    return () => controller.abort();
  }, [attempt]);
  if (!ready)
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 p-6">
        <p>{error || "正在连接 MonoPlanner…"}</p>
        {error && (
          <button
            className="rounded-lg border px-4 py-2"
            onClick={() => setAttempt(attempt + 1)}
          >
            重新连接
          </button>
        )}
      </div>
    );
  return <ServiceHealth.Provider value={storageError}>{children}</ServiceHealth.Provider>;
}
