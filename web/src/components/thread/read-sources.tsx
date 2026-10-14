"use client";

import { useEffect, useRef, useState } from "react";
import { FolderPlus, FilePlus, X } from "lucide-react";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

type Source = { path: string; kind: "file" | "directory" };

export function ReadSources({
  apiUrl,
  threadId,
  ensureThread,
  disabled,
}: {
  apiUrl: string;
  threadId: string | null;
  ensureThread: () => Promise<string>;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [sources, setSources] = useState<Source[]>([]);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const reads = useRef(0);
  const [error, setError] = useState("");
  const [revision, reconnect] = useState(0);
  const current = useRef(threadId);
  current.current = threadId;
  useEffect(() => {
    setSources([]);
    setError("");
    if (!open || !threadId) {
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    const version = ++reads.current;
    setLoading(true);
    fetch(`${apiUrl}/threads/${threadId}/read-sources`, {
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (response) => {
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "无法读取资料授权");
        if (!controller.signal.aborted && version === reads.current)
          setSources(result.sources);
      })
      .catch((e) => {
        if (!controller.signal.aborted && version === reads.current)
          setError(String(e));
      })
      .finally(() => {
        if (!controller.signal.aborted && version === reads.current)
          setLoading(false);
      });
    return () => controller.abort();
  }, [apiUrl, threadId, open, revision]);

  async function change(kind?: Source["kind"], remove?: Source) {
    if (busy || loading) return;
    reads.current++;
    setBusy(true);
    setError("");
    try {
      const id = await ensureThread();
      if (current.current !== id) throw new Error("对话已切换，请重新选择");
      const response = await fetch(
        `${apiUrl}/threads/${id}/${kind ? "choose-file" : "read-sources"}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            kind
              ? { kind }
              : { sources: sources.filter((s) => s.path !== remove?.path) },
          ),
        },
      );
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "资料授权未保存");
      if (current.current === id && result.sources) {
        reads.current++;
        setLoading(false);
        setSources(result.sources);
      }
    } catch (e) {
      if (current.current === threadId || !threadId)
        setError(e instanceof Error ? e.message : "资料授权未保存");
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Button
        type="button"
        variant="ghost"
        disabled={disabled}
        onClick={() => setOpen(true)}
        className="h-11 w-full justify-start gap-3 rounded-lg px-3 text-sm"
      >
        <FolderPlus className="size-4" />
        额外可读资料
      </Button>
      <Dialog
        open={open}
        onOpenChange={setOpen}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>额外可读资料</DialogTitle>
            <DialogDescription>
              文件只授权自身，文件夹授权其中资料。仅供读取，不授予修改、执行或启动软件的权限。
            </DialogDescription>
          </DialogHeader>
          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              disabled={busy || loading || disabled}
              onClick={() => void change("file")}
            >
              <FilePlus className="size-4" />
              选择文件
            </Button>
            <Button
              type="button"
              variant="outline"
              disabled={busy || loading || disabled}
              onClick={() => void change("directory")}
            >
              <FolderPlus className="size-4" />
              选择文件夹
            </Button>
          </div>
          <div className="max-h-64 space-y-2 overflow-y-auto">
            {!sources.length && (
              <p className="text-muted-foreground text-sm">
                {loading ? "正在读取资料授权…" : "尚未添加额外资料。"}
              </p>
            )}
            {sources.map((source) => (
              <div
                key={source.path}
                className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm"
              >
                <span className="text-muted-foreground shrink-0 text-xs">
                  {source.kind === "file" ? "文件" : "文件夹"}
                </span>
                <span className="min-w-0 flex-1 break-all">{source.path}</span>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  disabled={busy || loading || disabled}
                  aria-label={`移除 ${source.path}`}
                  onClick={() => void change(undefined, source)}
                >
                  <X className="size-4" />
                </Button>
              </div>
            ))}
          </div>
          {busy && (
            <p
              role="status"
              className="text-muted-foreground text-sm"
            >
              请完成选择窗口中的操作，或等待授权保存。
            </p>
          )}
          {error && (
            <div
              role="alert"
              className="text-destructive text-sm"
            >
              {error}
              <Button
                type="button"
                variant="ghost"
                onClick={() => reconnect((v) => v + 1)}
              >
                重新连接
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
