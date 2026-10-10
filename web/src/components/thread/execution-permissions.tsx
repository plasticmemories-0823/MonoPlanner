"use client";

import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

type Mode = "sandbox" | "ask" | "full";
type Permission = { requested: Mode; effective: Mode };
const sandbox: Permission = { requested: "sandbox", effective: "sandbox" };
const name = (mode: Mode) => ({ sandbox: "沙盒权限", ask: "风险询问", full: "完全权限" })[mode];

export function ExecutionPermissions({
  apiUrl,
  threadId,
  running,
  ensureThread,
}: {
  apiUrl: string;
  threadId: string | null;
  running: boolean;
  ensureThread: () => Promise<string>;
}) {
  const scope = `${apiUrl}\0${threadId || ""}`;
  const currentScope = useRef(scope);
  currentScope.current = scope;
  const revision = useRef(0);
  const [saved, setSaved] = useState({ scope, value: sandbox });
  const permission = saved.scope === scope ? saved.value : sandbox;
  const pending = permission.requested !== permission.effective;
  const [open, setOpen] = useState(false);
  const [confirmFull, setConfirmFull] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setConfirmFull(false);
    setError("");
  }, [scope]);
  useEffect(() => {
    if (!threadId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      const version = revision.current;
      try {
        const response = await fetch(
          `${apiUrl}/threads/${threadId}/permission`,
          {
            cache: "no-store",
            signal: controller.signal,
          },
        );
        const value = await response.json();
        if (!response.ok) throw new Error(value.error || "无法读取执行权限");
        if (
          !controller.signal.aborted &&
          currentScope.current === scope &&
          revision.current === version
        )
          setSaved({ scope, value });
        if (!controller.signal.aborted && running)
          timer = setTimeout(poll, 1500);
      } catch (error) {
        if (!controller.signal.aborted && revision.current === version)
          setError(error instanceof Error ? error.message : "无法读取执行权限");
      }
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [apiUrl, threadId, scope, running, pending]);

  async function change(mode: Mode) {
    if (busy) return;
    setBusy(true);
    setError("");
    revision.current += 1;
    try {
      const id = await ensureThread();
      const targetScope = `${apiUrl}\0${id}`;
      if (currentScope.current !== targetScope)
        throw new Error("对话已切换，请在当前对话重新操作");
      revision.current += 1;
      const response = await fetch(`${apiUrl}/threads/${id}/permission`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || "权限保存失败");
      revision.current += 1;
      if (currentScope.current === targetScope) {
        setSaved({ scope: targetScope, value });
        setConfirmFull(false);
        toast.success(
          value.requested === value.effective
            ? `当前权限：${name(value.effective)}`
            : running
              ? "权限选择已保存，当前操作结束后生效"
              : "权限选择已保存，下一次操作开始前生效",
        );
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "权限保存失败";
      setError(message);
      toast.error(message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        aria-label="执行权限"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen(true)}
        title={`当前：${name(permission.effective)}${pending ? `；待切换为${name(permission.requested)}` : ""}`}
        className="text-muted-foreground h-8 shrink-0 gap-1.5 rounded-full px-2 text-xs"
      >
        <ShieldCheck className="size-4" />
        {name(permission.effective)}
        {pending && (
          <span className="rounded bg-amber-100 px-1 text-[10px] text-amber-800">
            待切换
          </span>
        )}
        <ChevronDown className="size-3" />
      </Button>
      <Dialog
        open={open}
        onOpenChange={(next) => {
          setOpen(next);
          if (!next) setConfirmFull(false);
        }}
      >
        <DialogContent
          className="max-w-md"
          closeLabel="关闭权限选择"
        >
          <DialogHeader>
            <DialogTitle>执行权限</DialogTitle>
            <DialogDescription>
              仅作用于当前对话，由你选择；新对话默认沙盒。
            </DialogDescription>
          </DialogHeader>
          <div
            role="status"
            className="bg-muted/50 space-y-1 rounded-lg p-3 text-sm"
          >
            <p>
              当前有效权限：<strong>{name(permission.effective)}</strong>
            </p>
            <p>已选权限：{name(permission.requested)}</p>
            {pending && (
              <p className="text-amber-700">
                {running ? "当前操作结束后生效" : "下一次操作开始前生效"}
              </p>
            )}
          </div>
          <div className="grid gap-2">
            {(["sandbox", "ask", "full"] as const).map((mode) => (
              <Button
                key={mode}
                type="button"
                variant="outline"
                disabled={busy}
                aria-label={`选择${name(mode)}`}
                aria-pressed={permission.requested === mode}
                className="h-auto min-h-16 justify-start gap-3 p-3 text-left whitespace-normal"
                onClick={() =>
                  mode === "full" && permission.requested !== "full"
                    ? setConfirmFull(true)
                    : (setConfirmFull(false),
                      permission.requested !== mode && void change(mode))
                }
              >
                <span className="flex-1">
                  <span className="block font-medium">{name(mode)}</span>
                  <span className="text-muted-foreground mt-1 block text-xs leading-5 font-normal">
                    {mode === "sandbox"
                      ? "执行留在沙盒，允许受控预览；不启动外部软件，不写入目录外文件。"
                      : mode === "ask" ? "执行仍留在沙盒；外部打开及明确的目录外文件操作须逐次批准。" : "使用当前 Windows 用户权限，不逐次确认；不会申请管理员权限。"}
                  </span>
                </span>
                {permission.requested === mode && (
                  <Check className="size-4 shrink-0" />
                )}
              </Button>
            ))}
          </div>
          {confirmFull && (
            <div className="space-y-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950">
              <p>
                完全权限不提供目录隔离，也不会申请管理员权限。确认后，此对话持续使用该选择，直到你手动切回沙盒。
              </p>
              <div className="flex justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  disabled={busy}
                  onClick={() => setConfirmFull(false)}
                >
                  取消
                </Button>
                <Button
                  type="button"
                  disabled={busy}
                  onClick={() => void change("full")}
                >
                  确认开启完全权限
                </Button>
              </div>
            </div>
          )}
          {busy && (
            <p
              role="status"
              className="text-muted-foreground text-sm"
            >
              正在保存权限选择…
            </p>
          )}
          {error && (
            <p
              role="alert"
              className="text-destructive text-sm"
            >
              {error}；权限未更改，请核对后重试。
            </p>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
