import { cookies } from "next/headers";
import { readStorageFailure } from "@/lib/conversation-api";
import {
  allowedRequest,
  backendUrl,
  instanceToken,
  MAX_BODY,
  SESSION_COOKIE,
  validSession,
} from "@/lib/server-access";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

async function forward(request: Request, context: { params: Promise<{ path: string[] }> }) {
  const mutable = !["GET", "HEAD"].includes(request.method);
  if (!allowedRequest(request, mutable))
    return Response.json({ error: "Cross-site request denied" }, { status: 403 });
  try {
    if (!validSession((await cookies()).get(SESSION_COOKIE)?.value))
      return Response.json({ error: "请重新连接本地服务。" }, { status: 401 });
    const { path } = await context.params;
    if (
      path.some(
        (part) =>
          !part ||
          part === "." ||
          part === ".." ||
          /[\\/]/.test(part) ||
          part.includes(String.fromCharCode(0)),
      )
    )
      return Response.json({ error: "Invalid request path" }, { status: 400 });
    const controller = new AbortController();
    request.signal.addEventListener("abort", () => controller.abort(), {
      once: true,
    });
    let body: Uint8Array | undefined;
    if (mutable && request.body) {
      const reader = request.body.getReader();
      const chunks: Uint8Array[] = [];
      let size = 0;
      while (true) {
        const result = await reader.read();
        if (result.done) break;
        size += result.value.length;
        if (size > MAX_BODY) {
          await reader.cancel();
          return Response.json({ error: "Request too large" }, { status: 413 });
        }
        chunks.push(result.value);
      }
      if (
        size &&
        !request.headers.get("content-type")?.toLowerCase().startsWith("application/json")
      )
        return Response.json({ error: "JSON request required" }, { status: 415 });
      body = Buffer.concat(chunks);
    }
    const headers = new Headers({
      "X-MonoPlanner-Service": instanceToken(),
      Origin: new URL(request.url).origin,
    });
    for (const key of ["content-type", "accept", "last-event-id"])
      if (request.headers.has(key)) headers.set(key, request.headers.get(key)!);
    const upstream = await fetch(
      `${backendUrl()}/${path.map(encodeURIComponent).join("/")}${new URL(request.url).search}`,
      {
        method: request.method,
        headers,
        body: body as BodyInit | undefined,
        signal: controller.signal,
        redirect: "manual",
        cache: "no-store",
      },
    );
    const output = new Headers({
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    });
    for (const key of [
      "content-type",
      "content-disposition",
      "content-security-policy",
      "referrer-policy",
    ])
      if (upstream.headers.has(key)) output.set(key, upstream.headers.get(key)!);
    // 不向浏览器传递后端 Cookie、凭据、重定向地址或内部错误正文。
    if (upstream.status >= 500) {
      const value = await upstream.json().catch(() => null);
      const fault = readStorageFailure(value?.storage_error);
      if (fault)
        return Response.json(
          { error: fault.message, storage_error: fault },
          { status: upstream.status, headers: output },
        );
      return Response.json(
        { error: "本地服务暂时不可用；请检查启动日志后重新连接。" },
        { status: upstream.status },
      );
    }
    if (!upstream.body) return new Response(null, { status: upstream.status, headers: output });
    const reader = upstream.body.getReader();
    return new Response(
      new ReadableStream({
        async pull(target) {
          try {
            const next = await reader.read();
            if (next.done) target.close();
            else target.enqueue(next.value);
          } catch (error) {
            target.error(error);
          }
        },
        async cancel() {
          controller.abort();
          await reader.cancel().catch(() => {});
        },
      }),
      { status: upstream.status, headers: output },
    );
  } catch {
    return Response.json(
      { error: "无法连接本地服务，请重新连接；不会重发任务。" },
      { status: 503 },
    );
  }
}

export {
  forward as GET,
  forward as POST,
  forward as PUT,
  forward as PATCH,
  forward as DELETE,
  forward as HEAD,
};
