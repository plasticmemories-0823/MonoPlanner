import type { Thread } from "@langchain/langgraph-sdk";

export type Visibility = "active" | "archived" | "deleted";
export type ConversationTitle = {
  title: string;
  status: string;
  revision: number;
};
export type ManagedThread = Thread & {
  visibility: Visibility;
  title?: ConversationTitle;
};

function latestTitle(
  previous?: ConversationTitle,
  incoming?: ConversationTitle,
) {
  if (!incoming || (previous && previous.revision > incoming.revision))
    return previous;
  return incoming;
}

// 本地修改优先于进行中的搜索快照；标题是独立的
// 展示记录，不会成为消息或任务输入。
export class ThreadHistoryStore {
  rows: ManagedThread[] = [];
  private pending = new Set<string>();
  private purged = new Set<string>();
  private visibility = new Map<string, { value: Visibility; after: number }>();
  private createdHere = new Set<string>();
  private firstSent = new Set<string>();
  private waitingTitles = new Map<string, number>();
  private pausedTitles = new Set<string>();
  private nextRefresh = 0;
  private appliedRefresh = 0;

  startRefresh() {
    return ++this.nextRefresh;
  }

  applyRefresh(
    ticket: number,
    incoming: ManagedThread[],
    visibility: Record<string, string> = {},
  ) {
    if (ticket < this.appliedRefresh) return this.rows;
    this.appliedRefresh = ticket;
    const old = new Map(this.rows.map((row) => [row.thread_id, row]));
    for (const [id, value] of Object.entries(visibility))
      if (value === "purged") this.purged.add(id);
    const found = new Set(incoming.map((row) => row.thread_id));
    const rows: ManagedThread[] = incoming
      .filter((row) => !this.purged.has(row.thread_id))
      .map((row) => {
        const id = row.thread_id;
        const override = this.visibility.get(id);
        if (override?.value === row.visibility && ticket > override.after)
          this.visibility.delete(id);
        this.pending.delete(id);
        return {
          ...row,
          values: (row.values as { messages?: unknown[] })?.messages?.length
            ? row.values
            : old.get(id)?.values || row.values,
          title: latestTitle(old.get(id)?.title, row.title),
          visibility: override?.value || row.visibility,
        };
      });
    // 新建会话可能尚未出现在搜索结果中。
    for (const row of this.rows)
      if (
        !found.has(row.thread_id) &&
        this.pending.has(row.thread_id) &&
        !this.purged.has(row.thread_id)
      )
        rows.push(row);
    this.rows = this.sort(rows);
    for (const row of this.rows) this.trackTitle(row.thread_id);
    return this.rows;
  }

  upsert(thread: Thread, firstMessage?: string) {
    const id = thread.thread_id;
    if (this.purged.has(id)) return this.rows;
    this.createdHere.add(id);
    const previous = this.rows.find((row) => row.thread_id === id);
    this.pending.add(id);
    const values = (thread.values as { messages?: unknown[] })?.messages?.length
      ? thread.values
      : previous?.values || thread.values;
    const row: ManagedThread = {
      ...previous,
      ...thread,
      values:
        firstMessage && !(values as { messages?: unknown[] })?.messages?.length
          ? { ...values, messages: [{ type: "human", content: firstMessage }] }
          : values,
      visibility:
        this.visibility.get(id)?.value || previous?.visibility || "active",
    };
    this.rows = this.sort([
      row,
      ...this.rows.filter((item) => item.thread_id !== id),
    ]);
    return this.rows;
  }

  applyTitle(id: string, title: ConversationTitle) {
    const previous = this.rows.find((row) => row.thread_id === id);
    if (!previous || this.purged.has(id)) return this.rows;
    const next = latestTitle(previous.title, title);
    if (JSON.stringify(next) === JSON.stringify(previous.title))
      return this.rows;
    this.rows = this.rows.map((row) =>
      row.thread_id === id ? { ...row, title: next } : row,
    );
    this.trackTitle(id);
    return this.rows;
  }

  waitForTitle(id: string, text: string, now = Date.now()) {
    if (!this.createdHere.has(id) || this.firstSent.has(id)) return this.rows;
    const row = this.rows.find((item) => item.thread_id === id);
    if (!row || this.purged.has(id)) return this.rows;
    this.firstSent.add(id);
    if (!row.title) this.waitingTitles.set(id, now);
    if (!(row.values as { messages?: unknown[] })?.messages?.length) {
      this.rows = this.rows.map((item) =>
        item.thread_id === id
          ? {
              ...item,
              values: {
                ...item.values,
                messages: [{ type: "human", content: text }],
              },
            }
          : item,
      );
    }
    return this.rows;
  }

  applyTitles(titles: Record<string, ConversationTitle>) {
    for (const [id, title] of Object.entries(titles))
      this.applyTitle(id, title);
    return this.rows;
  }

  titleRefreshState(now = Date.now()) {
    let expired = false;
    for (const [id, started] of this.waitingTitles) {
      const row = this.rows.find((item) => item.thread_id === id);
      if (
        !row ||
        this.purged.has(id) ||
        (row.title && !this.titlePending(row.title))
      ) {
        this.waitingTitles.delete(id);
      } else if (now - started >= (row.title ? 120_000 : 30_000)) {
        this.waitingTitles.delete(id);
        this.pausedTitles.add(id);
        expired = true;
      }
    }
    return { pending: this.waitingTitles.size > 0, expired };
  }

  retryTitles(now = Date.now()) {
    for (const id of this.pausedTitles) this.waitingTitles.set(id, now);
    this.pausedTitles.clear();
  }

  private titlePending(title: ConversationTitle) {
    return title.status === "pending" || title.status === "running";
  }

  private trackTitle(id: string) {
    const title = this.rows.find((row) => row.thread_id === id)?.title;
    if (!title) return;
    if (this.titlePending(title)) {
      if (!this.waitingTitles.has(id) && !this.pausedTitles.has(id))
        this.waitingTitles.set(id, Date.now());
    } else {
      this.waitingTitles.delete(id);
      this.pausedTitles.delete(id);
    }
  }

  setVisibility(id: string, value: Visibility) {
    this.visibility.set(id, { value, after: this.nextRefresh });
    this.rows = this.rows.map((row) =>
      row.thread_id === id ? { ...row, visibility: value } : row,
    );
    return this.rows;
  }

  purge(id: string) {
    this.purged.add(id);
    this.pending.delete(id);
    this.visibility.delete(id);
    this.waitingTitles.delete(id);
    this.pausedTitles.delete(id);
    this.createdHere.delete(id);
    this.firstSent.delete(id);
    this.rows = this.rows.filter((row) => row.thread_id !== id);
    return this.rows;
  }

  private sort(rows: ManagedThread[]) {
    return rows.sort((a, b) => b.created_at.localeCompare(a.created_at));
  }
}
