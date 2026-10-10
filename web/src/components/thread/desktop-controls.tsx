"use client";

import { useEffect, useState } from "react";
import { AppWindow, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";

type Permission = {
  requested: "sandbox" | "full";
  effective: "sandbox" | "full";
};
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

export function DesktopControls({
  apiUrl,
  threadId,
  running,
}: {
  apiUrl: string;
  threadId: string | null;
  running: boolean;
}) {
  const [permission, setPermission] = useState<Permission>({
    requested: "sandbox",
    effective: "sandbox",
  });
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    setPermission({ requested: "sandbox", effective: "sandbox" });
    if (!threadId) return;
    const poll = async () => {
      try {
        const result = await jsonRequest(
          `${apiUrl}/threads/${threadId}/permission`,
        );
        if (!cancelled) setPermission(result);
      } catch (error) {
        if (!cancelled) toast.error(String(error));
      }
      if (!cancelled && running) timer = setTimeout(poll, 1500);
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [apiUrl, threadId, running]);

  async function change(mode: string) {
    if (!threadId) return;
    if (
      mode === "full" &&
      !window.confirm(
        "完全访问使用你的 Windows 用户权限，可写入工作目录外及访问网络。仅作用于当前对话，持续到你手动关闭。是否开启？",
      )
    )
      return;
    setBusy(true);
    try {
      setPermission(
        await jsonRequest(`${apiUrl}/threads/${threadId}/permission`, { mode }),
      );
    } catch (error) {
      toast.error(String(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <label
      className="text-muted-foreground flex h-8 shrink-0 items-center gap-1 rounded-full px-2 text-xs"
      title={
        permission.requested !== permission.effective
          ? `权限待生效；当前为${permission.effective === "full" ? "完全访问" : "沙盒"}`
          : "新对话默认沙盒；权限只由你切换，在下一个安全边界生效。"
      }
    >
      <ShieldCheck className="size-4" />
      <select
        aria-label="执行权限"
        value={permission.requested}
        disabled={!threadId || busy}
        onChange={(e) => change(e.target.value)}
        className="max-w-28 rounded-md bg-transparent p-1"
      >
        <option value="sandbox">沙盒</option>
        <option value="full">完全访问</option>
      </select>
      {permission.requested !== permission.effective && (
        <span
          className="size-1.5 rounded-full bg-amber-500"
          aria-label="权限待生效"
        />
      )}
    </label>
  );
}

export function DesktopApplicationSettings({
  apiUrl,
  threadId,
}: {
  apiUrl: string;
  threadId: string | null;
}) {
  const [busy, setBusy] = useState(false);
  async function choose(application = true) {
    if (!threadId) return;
    setBusy(true);
    try {
      await jsonRequest(
        `${apiUrl}/threads/${threadId}/${application ? "choose-application" : "choose-file"}`,
        {},
      );
      if (application)
        toast.success("应用已登记，可以在聊天中请求启动或打开文件");
      else window.dispatchEvent(new Event("monoplanner-opens"));
    } catch (error) {
      toast.error(String(error));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="flex items-center justify-between gap-3 border-t pt-3">
      <p className="text-muted-foreground text-sm">
        仅授权选中的文件，或登记用于打开文件的软件。
      </p>
      <div className="flex shrink-0 gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={!threadId || busy}
          onClick={() => void choose(false)}
        >
          授权文件
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={!threadId || busy}
          onClick={() => void choose()}
        >
          <AppWindow className="size-4" />
          登记应用
        </Button>
      </div>
    </div>
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
