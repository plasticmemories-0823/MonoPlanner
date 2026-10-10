import { Button } from "@/components/ui/button";
import { useThreads, ManagedThread, Visibility } from "@/providers/Thread";
import { useEffect, useRef, useState } from "react";
import { getContentString } from "../utils";
import { useQueryState, parseAsBoolean } from "nuqs";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { MoreHorizontal, PanelLeftClose } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";

function HistoryList({ close }: { close?: () => void }) {
  const { threads, threadsLoading, setVisibility, deletePermanently } =
    useThreads();
  const [confirmDelete, setConfirmDelete] = useState<ManagedThread>();
  const [deleting, setDeleting] = useState(false);
  const [threadId, setThreadId] = useQueryState("threadId");
  const [view, setView] = useState<Visibility>("active");
  const [menu, setMenu] = useState<{
    thread: ManagedThread;
    x: number;
    y: number;
  }>();
  const menuRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!menu) return;
    const dismiss = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenu(undefined);
    };
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenu(undefined);
    };
    document.addEventListener("pointerdown", dismiss);
    document.addEventListener("keydown", key);
    menuRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
    return () => {
      document.removeEventListener("pointerdown", dismiss);
      document.removeEventListener("keydown", key);
    };
  }, [menu]);
  const change = async (visibility: Visibility) => {
    if (!menu) return;
    try {
      await setVisibility(menu.thread.thread_id, visibility);
      if (visibility !== "active" && threadId === menu.thread.thread_id)
        setThreadId(null);
      toast.success(
        visibility === "archived"
          ? "对话已归档"
          : visibility === "deleted"
            ? "已移入回收站，可随时恢复"
            : "对话已恢复",
      );
      setMenu(undefined);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "操作未完成");
    }
  };
  const showMenu = (thread: ManagedThread, x: number, y: number) =>
    setMenu({
      thread,
      x: Math.max(8, Math.min(x, window.innerWidth - 192)),
      y: Math.max(8, Math.min(y, window.innerHeight - 110)),
    });
  const remove = async () => {
    if (!confirmDelete || deleting) return;
    setDeleting(true);
    try {
      await deletePermanently(confirmDelete.thread_id);
      if (threadId === confirmDelete.thread_id) setThreadId(null);
      setConfirmDelete(undefined);
      toast.success("对话已彻底删除，无法恢复");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "彻底删除未完成");
    } finally {
      setDeleting(false);
    }
  };
  const visible = threads.filter((thread) => thread.visibility === view);
  return (
    <>
      <div
        role="tablist"
        aria-label="对话分类"
        className="flex w-full gap-1 px-2"
      >
        {(
          [
            ["active", "对话"],
            ["archived", "归档"],
            ["deleted", "回收站"],
          ] as const
        ).map(([value, label]) => (
          <Button
            role="tab"
            aria-selected={view === value}
            key={value}
            size="sm"
            variant={view === value ? "secondary" : "ghost"}
            className="h-8 flex-1 px-1 text-xs"
            onClick={() => {
              setView(value);
              setMenu(undefined);
            }}
          >
            {label}
          </Button>
        ))}
      </div>
      <div className="min-h-0 w-full flex-1 space-y-1 overflow-y-auto px-2 pb-3">
        {threadsLoading ? (
          Array.from({ length: 8 }, (_, i) => (
            <Skeleton
              key={i}
              className="h-9 w-full"
            />
          ))
        ) : !visible.length ? (
          <p className="text-muted-foreground px-2 py-4 text-xs">
            {view === "active"
              ? "暂无对话"
              : view === "archived"
                ? "暂无归档对话"
                : "回收站为空"}
          </p>
        ) : (
          visible.map((thread) => {
            const values = thread.values as
              { messages?: { content: any }[] } | undefined;
            const title = values?.messages?.[0]
              ? getContentString(values.messages[0].content)
              : "新对话";
            return (
              <div
                key={thread.thread_id}
                data-testid="history-item"
                className="group hover:bg-muted flex min-w-0 items-center rounded-lg"
                onContextMenu={(event) => {
                  event.preventDefault();
                  showMenu(thread, event.clientX, event.clientY);
                }}
              >
                <Button
                  variant="ghost"
                  className="min-w-0 flex-1 justify-start px-2 text-left font-normal"
                  title={title}
                  onClick={() => {
                    if (thread.visibility === "deleted") {
                      toast.info("请先从回收站恢复这条对话");
                      return;
                    }
                    setThreadId(thread.thread_id);
                    close?.();
                  }}
                >
                  <span className="truncate">{title}</span>
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-7 shrink-0 opacity-0 group-hover:opacity-100 focus:opacity-100"
                  aria-label={`管理对话：${title}`}
                  onClick={(event) => {
                    const rect = event.currentTarget.getBoundingClientRect();
                    showMenu(thread, rect.left, rect.bottom);
                  }}
                >
                  <MoreHorizontal className="size-4" />
                </Button>
              </div>
            );
          })
        )}
      </div>
      {menu && (
        <div
          ref={menuRef}
          role="menu"
          aria-label="对话管理"
          className="bg-background fixed z-50 w-44 rounded-xl border p-1 shadow-lg"
          style={{ left: menu.x, top: menu.y }}
          onKeyDown={(event) => {
            const buttons = [
              ...event.currentTarget.querySelectorAll<HTMLButtonElement>(
                "button",
              ),
            ];
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
              event.preventDefault();
              const index = buttons.indexOf(
                document.activeElement as HTMLButtonElement,
              );
              buttons[
                (index +
                  (event.key === "ArrowDown" ? 1 : -1) +
                  buttons.length) %
                  buttons.length
              ]?.focus();
            }
          }}
        >
          {menu.thread.visibility !== "active" && (
            <Button
              role="menuitem"
              variant="ghost"
              className="w-full justify-start"
              onClick={() => change("active")}
            >
              恢复对话
            </Button>
          )}
          {menu.thread.visibility === "active" && (
            <Button
              role="menuitem"
              variant="ghost"
              className="w-full justify-start"
              onClick={() => change("archived")}
            >
              归档
            </Button>
          )}
          {menu.thread.visibility !== "deleted" && (
            <Button
              role="menuitem"
              variant="ghost"
              className="text-destructive w-full justify-start"
              onClick={() => change("deleted")}
            >
              删除（可恢复）
            </Button>
          )}
          {menu.thread.visibility === "deleted" && (
            <Button
              role="menuitem"
              variant="ghost"
              className="text-destructive w-full justify-start"
              onClick={() => {
                setConfirmDelete(menu.thread);
                setMenu(undefined);
              }}
            >
              彻底删除
            </Button>
          )}
        </div>
      )}
      <Dialog
        open={!!confirmDelete}
        onOpenChange={(open) => {
          if (!open && !deleting) setConfirmDelete(undefined);
        }}
      >
        <DialogContent closeLabel="关闭删除确认">
          <DialogHeader>
            <DialogTitle>彻底删除这条对话？</DialogTitle>
            <DialogDescription>
              对话消息、思考内容和关联任务记录将被删除，无法恢复。工作仓库中的文件和独立评测记录会保留。
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              disabled={deleting}
              onClick={() => setConfirmDelete(undefined)}
            >
              取消
            </Button>
            <Button
              variant="destructive"
              disabled={deleting}
              onClick={remove}
            >
              {deleting ? "正在删除…" : "彻底删除"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default function ThreadHistory() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const { getThreads, setThreads, setThreadsLoading } = useThreads();
  useEffect(() => {
    let cancelled = false;
    setThreadsLoading(true);
    getThreads()
      .then((threads) => {
        if (!cancelled) setThreads(threads);
      })
      .catch(() => {
        if (!cancelled) toast.error("无法读取对话记录，请检查服务连接");
      })
      .finally(() => {
        if (!cancelled) setThreadsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [getThreads, setThreads, setThreadsLoading]);
  return (
    <>
      <div className="bg-muted/20 hidden h-screen w-full flex-col gap-3 lg:flex">
        <div className="flex h-16 w-full items-center justify-between px-3">
          <h1 className="text-sm font-semibold">对话记录</h1>
          <Button
            aria-label="收起对话记录"
            size="icon"
            variant="ghost"
            onClick={() => setChatHistoryOpen(false)}
          >
            <PanelLeftClose className="size-4" />
          </Button>
        </div>
        <HistoryList />
      </div>
      <Sheet
        open={!!chatHistoryOpen && !isLargeScreen}
        onOpenChange={(open) => {
          if (!isLargeScreen) setChatHistoryOpen(open);
        }}
      >
        <SheetContent
          side="left"
          className="flex w-72 flex-col gap-3 lg:hidden"
        >
          <SheetHeader>
            <SheetTitle>对话记录</SheetTitle>
          </SheetHeader>
          <HistoryList close={() => setChatHistoryOpen(false)} />
        </SheetContent>
      </Sheet>
    </>
  );
}
