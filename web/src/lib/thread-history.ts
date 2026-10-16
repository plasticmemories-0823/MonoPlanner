import type { Conversation as Thread } from "./conversation-types";

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

function latestTitle(previous?: ConversationTitle, incoming?: ConversationTitle) {
  if (!incoming || (previous && previous.revision > incoming.revision)) return previous;
  return incoming;
}

// 本地修改优先于进行中的搜索快照；标题是独立的
// 展示记录，不会成为消息或任务输入。
export class ThreadHistoryStore {
  rows: ManagedThread[] = [];
  private pending = new Set<string>();
  private purged = new Set<string>();
  private visibility = new Map<string, { value: Visibility; after: number }>();
  private nextRefresh = 0;
  private appliedRefresh = 0;

  startRefresh() {
    return ++this.nextRefresh;
  }

  applyRefresh(ticket: number, incoming: ManagedThread[]) {
    if (ticket < this.appliedRefresh) return this.rows;
    this.appliedRefresh = ticket;
    const old = new Map(this.rows.map((row) => [row.thread_id, row]));
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
    return this.rows;
  }

  upsert(thread: Thread, firstMessage?: string) {
    const id = thread.thread_id;
    if (this.purged.has(id)) return this.rows;
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
      visibility: this.visibility.get(id)?.value || previous?.visibility || "active",
    };
    this.rows = this.sort([row, ...this.rows.filter((item) => item.thread_id !== id)]);
    return this.rows;
  }

  applyTitle(id: string, title: ConversationTitle) {
    const previous = this.rows.find((row) => row.thread_id === id);
    if (!previous || this.purged.has(id)) return this.rows;
    const next = latestTitle(previous.title, title);
    if (JSON.stringify(next) === JSON.stringify(previous.title)) return this.rows;
    this.rows = this.rows.map((row) => (row.thread_id === id ? { ...row, title: next } : row));
    return this.rows;
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
    this.rows = this.rows.filter((row) => row.thread_id !== id);
    return this.rows;
  }

  private sort(rows: ManagedThread[]) {
    return rows.sort((a, b) => b.created_at.localeCompare(a.created_at));
  }
}
