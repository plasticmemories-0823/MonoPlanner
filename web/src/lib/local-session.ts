/** 重连只建立浏览器会话并检查本机服务，不重发聊天或任务请求。 */
export async function connectLocalSession(
  signal: AbortSignal,
  timeout = 8000,
): Promise<{ storage_error?: unknown }> {
  const bounded = AbortSignal.any([signal, AbortSignal.timeout(timeout)]);
  try {
    const session = await fetch("/session", {
      method: "POST",
      signal: bounded,
    });
    if (!session.ok) throw new Error("无法建立本机会话，请检查启动器是否正常运行。");
    const service = await fetch("/api/service", {
      cache: "no-store",
      signal: bounded,
    });
    if (!service.ok) throw new Error("本机服务暂未就绪，请重新连接。");
    return service.json();
  } catch (cause) {
    if (bounded.aborted && !signal.aborted)
      throw new Error("连接本机服务超时，请重新连接。", { cause });
    throw cause;
  }
}
