"use client";

import { useEffect, useState } from "react";
import { AppWindow, FilePlus } from "lucide-react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";

type OpenRecord = {
  id: string;
  kind: string;
  status: string;
  path: string;
  target?: string;
  application?: { name: string };
  error?: string;
};

async function jsonRequest(url: string, value?: object) {
  const response = await fetch(
    url,
    value
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(value),
        }
      : { cache: "no-store" },
  );
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "操作未完成");
  return result;
}

export function DesktopActions({
  apiUrl,
  ensureThread,
}: {
  apiUrl: string;
  ensureThread: () => Promise<string>;
}) {
  const [busy, setBusy] = useState(false);
  async function choose(application: boolean) {
    if (busy) return;
    setBusy(true);
    try {
      const thread = await ensureThread();
      const result = await jsonRequest(
        `${apiUrl}/threads/${thread}/${application ? "choose-application" : "choose-file"}`,
        {},
      );
      if (application && result.application)
        toast.success("本机应用已登记；可以在聊天中请求启动或打开文件");
      else if (!application && result.path)
        window.dispatchEvent(new Event("monoplanner-opens"));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "操作未完成");
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Button
        type="button"
        variant="ghost"
        disabled={busy}
        onClick={() => void choose(false)}
        className="h-11 w-full justify-start gap-3 rounded-lg px-3 text-sm"
      >
        <FilePlus className="size-4" />
        授权文件
      </Button>
      <Button
        type="button"
        variant="ghost"
        disabled={busy}
        onClick={() => void choose(true)}
        className="h-11 w-full justify-start gap-3 rounded-lg px-3 text-sm"
      >
        <AppWindow className="size-4" />
        登记本机应用
      </Button>
      {busy && (
        <p
          role="status"
          className="text-muted-foreground px-3 text-xs"
        >
          请在选择窗口中完成选择或取消。
        </p>
      )}
    </>
  );
}

export function DesktopCards({
  apiUrl,
  threadId,
  running,
}: {
  apiUrl: string;
  threadId: string | null;
  running: boolean;
}) {
  const [items, setItems] = useState<OpenRecord[]>([]);
  const [preview, setPreview] = useState<OpenRecord | null>(null);
  const [busy, setBusy] = useState("");
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    setItems([]);
    setPreview(null);
    if (!threadId) return;
    const poll = async () => {
      try {
        const result = await jsonRequest(`${apiUrl}/threads/${threadId}/opens`);
        if (!cancelled) setItems(result.items);
      } catch (error) {
        if (!cancelled) toast.error(String(error));
      }
      if (!cancelled && running) timer = setTimeout(poll, 1000);
    };
    const refresh = () => {
      clearTimeout(timer);
      void poll();
    };
    poll();
    window.addEventListener("monoplanner-opens", refresh);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      window.removeEventListener("monoplanner-opens", refresh);
    };
  }, [apiUrl, threadId, running]);

  async function open(item: OpenRecord) {
    setBusy(item.id);
    try {
      const result = await jsonRequest(`${apiUrl}/threads/${threadId}/opens`, {
        id: item.id,
        target: item.target,
      });
      setItems((old) => old.map((r) => (r.id === item.id ? result : r)));
    } catch (error) {
      toast.error(String(error));
    } finally {
      setBusy("");
    }
  }

  return (
    <>
      {items.map((item) => (
        <div
          key={item.id}
          className="border-border my-3 rounded-xl border p-3 text-sm"
          data-testid="desktop-open-card"
        >
          <p className="break-all">
            {item.kind === "preview" ? "文件预览" : item.application?.name} ·{" "}
            {item.path || "启动应用"}
          </p>
          {item.kind === "preview" ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setPreview(item)}
            >
              预览
            </Button>
          ) : item.status === "awaiting_approval" ? (
            <>
              <p className="text-muted-foreground my-2 text-xs">
                仅授权这一次打开。软件按正常用户权限运行，任务仍使用当前权限。
              </p>
              <Button
                type="button"
                size="sm"
                disabled={busy === item.id}
                onClick={() => open(item)}
              >
                打开
              </Button>
            </>
          ) : (
            <p className="text-muted-foreground mt-2 text-xs">
              {item.status === "opened"
                ? "已打开"
                : item.status === "launching"
                  ? "打开结果待核对，不会重复启动"
                  : item.error || "打开未完成"}
            </p>
          )}
        </div>
      ))}
      <Dialog
        open={!!preview}
        onOpenChange={(value) => {
          if (!value) setPreview(null);
        }}
      >
        <DialogContent className="h-[80vh] max-w-[90vw] grid-rows-[auto_minmax(0,1fr)] sm:max-w-5xl">
          <DialogHeader>
            <DialogTitle className="text-sm break-all">
              {preview?.path}
            </DialogTitle>
          </DialogHeader>
          {preview && (
            <iframe
              title="隔离文件预览"
              sandbox=""
              referrerPolicy="no-referrer"
              className="h-full min-h-0 w-full rounded-md border"
              src={`${apiUrl}/threads/${threadId}/preview/${preview.id}`}
            />
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
