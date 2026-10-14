"use client";

import { useEffect, useState } from "react";
import { requestJson } from "@/lib/conversation-api";
import type { Activity } from "@/lib/conversation-types";
import { appendDiagnosticPage } from "@/lib/work-diagnostics";
import { ActivityRows } from "./activity";
import { Button } from "../ui/button";

/** 工作原件仅在用户展开时读取，不参与主聊天、摘要或自动轮询。 */
export function WorkDiagnostics({ apiUrl, threadId }: { apiUrl: string; threadId: string }) {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<Activity[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState({ cursors: {} as Record<string, number>, more: false });
  const [query, setQuery] = useState({
    after: {} as Record<string, number>,
    replace: true,
    revision: 0,
  });
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    const parameters = new URLSearchParams({ after: JSON.stringify(query.after) });
    void requestJson<{
      items: Activity[];
      live?: Activity[];
      cursors: Record<string, number>;
      more: boolean;
    }>(`${apiUrl}/conversations/${threadId}/diagnostics?${parameters}`, {
      signal: controller.signal,
    })
      .then((result) => {
        if (!controller.signal.aborted) {
          setItems((old) =>
            appendDiagnosticPage(query.replace ? [] : old, [
              ...result.items,
              ...(result.live || []),
            ]),
          );
          setPage({ cursors: result.cursors || {}, more: Boolean(result.more) });
        }
      })
      .catch((failure) => {
        if (!controller.signal.aborted)
          setError(failure instanceof Error ? failure.message : "工作记录暂时不可用。");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [apiUrl, threadId, open, query]);
  return (
    <section className="space-y-2">
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => {
            if (!open)
              setQuery((old) => ({ after: {}, replace: true, revision: old.revision + 1 }));
            setOpen((value) => !value);
          }}
          aria-expanded={open}
        >
          {open ? "收起工作记录" : "查看工作记录"}
        </Button>
        {open && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={loading}
            onClick={() =>
              setQuery((old) => ({ after: {}, replace: true, revision: old.revision + 1 }))
            }
          >
            刷新工作记录
          </Button>
        )}
      </div>
      {open && (
        <>
          {loading && (
            <p
              role="status"
              className="text-muted-foreground text-xs"
            >
              正在读取已保存的工作记录…
            </p>
          )}
          {error && (
            <p
              role="alert"
              className="text-destructive text-sm"
            >
              {error}
            </p>
          )}
          {!loading && !error && !items.length && (
            <p className="text-muted-foreground text-xs">暂无可展示的工作记录。</p>
          )}
          <ActivityRows items={items} />
          {page.more && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={loading}
              onClick={() =>
                setQuery((old) => ({
                  after: page.cursors,
                  replace: false,
                  revision: old.revision + 1,
                }))
              }
            >
              加载更多工作记录
            </Button>
          )}
        </>
      )}
    </section>
  );
}
