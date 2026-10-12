"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";

type OpenRecord = {
  id: string; kind: string; status: string; path: string; target?: string;
  application?: { name: string }; error?: string; operation?: string; destination?: string;
  diff?: string; backup?: string;
};
const names: Record<string, string> = { create: "创建文件", replace: "替换文件", copy: "复制文件", move: "移动文件", delete: "删除文件" };

async function jsonRequest(url: string, value?: object, signal?: AbortSignal) {
  const response = await fetch(url, value ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value), signal,
  } : { cache: "no-store", signal });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "操作未完成");
  return result;
}

export function DesktopCards({ apiUrl, threadId, running }: {
  apiUrl: string; threadId: string | null; running: boolean;
}) {
  const scope = `${apiUrl}\0${threadId || ""}`;
  const current = useRef(scope);
  current.current = scope;
  const revision = useRef(0);
  const [view, setView] = useState<{ scope: string; items: OpenRecord[] }>({ scope, items: [] });
  const items = view.scope === scope ? view.items : [];
  const [preview, setPreview] = useState<OpenRecord | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [retry, reconnect] = useState(0);
  useEffect(() => {
    setPreview(null);
    setError("");
    if (!threadId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let active = false, requested = false;
    const poll = async () => {
      if (active) { requested = true; return; }
      active = true;
      const version = revision.current;
      try {
        const result = await jsonRequest(`${apiUrl}/threads/${threadId}/opens`, undefined, controller.signal);
        if (!controller.signal.aborted && current.current === scope && revision.current === version) {
          setView({ scope, items: result.items });
          setError("");
        }
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "无法读取操作记录");
      } finally {
        active = false;
        if (!controller.signal.aborted && (running || requested)) {
          const immediate = requested;
          requested = false;
          timer = setTimeout(poll, immediate ? 0 : 1000);
        }
      }
    };
    const refresh = () => { clearTimeout(timer); void poll(); };
    void poll();
    window.addEventListener("monoplanner-opens", refresh);
    return () => { controller.abort(); clearTimeout(timer); window.removeEventListener("monoplanner-opens", refresh); };
  }, [apiUrl, threadId, running, scope, retry]);

  async function decide(item: OpenRecord, action: "approve" | "reject") {
    if (busy) return;
    setBusy(item.id); setError(""); revision.current += 1;
    try {
      const result = await jsonRequest(`${apiUrl}/threads/${threadId}/opens`, { id: item.id, target: item.target, action });
      revision.current += 1;
      if (current.current === scope) setView(old => ({ scope, items: old.items.map(r => r.id === item.id ? result : r) }));
    } catch (e) {
      if (current.current === scope) setError(e instanceof Error ? e.message : "操作未完成");
    } finally { setBusy(""); }
  }
  return <>
    {error && <div role="alert" className="text-destructive text-sm">{error}<Button type="button" variant="ghost" onClick={() => reconnect(v => v + 1)}>重新连接</Button></div>}
    {items.map(item => <div key={item.id} className="border-border my-3 rounded-xl border p-3 text-sm" data-testid="desktop-open-card">
      <p className="break-all">{item.kind === "preview" ? "文件预览" : item.kind === "file_operation" ? names[item.operation || ""] : item.application?.name} · {item.path || "启动应用"}</p>
      {item.destination && item.destination !== item.path && <p className="text-muted-foreground mt-1 break-all">目标：{item.destination}</p>}
      {item.diff && <details className="my-2"><summary className="cursor-pointer">查看内容变更</summary><pre className="max-h-64 overflow-auto whitespace-pre-wrap">{item.diff}</pre></details>}
      {item.kind === "preview" ? <Button type="button" variant="outline" size="sm" onClick={() => setPreview(item)}>预览</Button>
        : item.status === "awaiting_approval" ? <>
          <p className="text-muted-foreground my-2 text-xs">仅批准所列目标这一次操作，不改变后续执行权限。{item.kind === "external" ? "外部软件按正常 Windows 用户权限运行。" : "不会授予模型代码目录外执行权限。"}</p>
          <div className="flex gap-2">
            <Button type="button" size="sm" disabled={!!busy} onClick={() => void decide(item, "approve")}>{item.kind === "external" ? "打开一次" : "批准一次"}</Button>
            <Button type="button" variant="outline" size="sm" disabled={!!busy} onClick={() => void decide(item, "reject")}>拒绝</Button>
          </div>
        </> : <p className="text-muted-foreground mt-2 text-xs">{item.status === "opened" ? "已打开" : item.status === "completed" ? "操作已完成" : item.status === "rejected" ? "已拒绝，未执行" : ["launching", "executing", "unknown"].includes(item.status) ? "结果待核对，不会重复执行" : item.error || "操作未完成"}</p>}
      {item.backup && <a className="mt-2 inline-block underline" href={`${apiUrl}/threads/${threadId}/opens/${item.id}/backup`}>下载恢复副本</a>}
    </div>)}
    <Dialog open={!!preview} onOpenChange={value => { if (!value) setPreview(null); }}>
      <DialogContent className="h-[80vh] max-w-[90vw] grid-rows-[auto_minmax(0,1fr)] sm:max-w-5xl">
        <DialogHeader><DialogTitle className="text-sm break-all">{preview?.path}</DialogTitle></DialogHeader>
        {preview && <iframe title="隔离文件预览" sandbox="" referrerPolicy="no-referrer" className="h-full min-h-0 w-full rounded-md border" src={`${apiUrl}/threads/${threadId}/preview/${preview.id}`} />}
      </DialogContent>
    </Dialog>
  </>;
}
