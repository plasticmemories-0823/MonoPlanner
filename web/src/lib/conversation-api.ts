import type { ConversationSnapshot } from "./conversation-types";

export async function requestJson<T>(
  url: string,
  init: RequestInit = {},
  fetcher: typeof fetch = fetch,
): Promise<T> {
  const signal = init.signal
    ? AbortSignal.any([init.signal, AbortSignal.timeout(15_000)])
    : AbortSignal.timeout(15_000);
  const response = await fetcher(url, { ...init, signal, cache: "no-store" });
  const value = await response.json();
  if (!response.ok)
    throw new Error(value.error || value.detail || "本地服务暂时不可用，请保留输入并重新连接。");
  return value as T;
}

type EventConnection = {
  addEventListener(type: string, listener: (event: MessageEvent) => void): void;
  onerror: ((event: Event) => void) | null;
  close(): void;
};
type Observer = {
  snapshot(value: ConversationSnapshot): void;
  error(message: string): void;
};

/** 一个会话只有一条事件连接和一个快照读取；事件从不直接拼接聊天内容。 */
export class ConversationMonitor {
  private connection?: EventConnection;
  private controller?: AbortController;
  private pending?: Promise<void>;
  private dirty = false;
  private disposed = false;
  private revision = 0;
  private cursor = 0;
  private appliedCursor = -1;
  private readOnly = false;
  private timer?: ReturnType<typeof setTimeout>;
  private apiUrl: string;
  private threadId: string;
  private observer: Observer;
  private fetcher: typeof fetch;
  private connect: (url: string) => EventConnection;

  constructor(
    apiUrl: string,
    threadId: string,
    observer: Observer,
    fetcher: typeof fetch = fetch,
    connect: (url: string) => EventConnection = (url) => new EventSource(url),
  ) {
    this.apiUrl = apiUrl;
    this.threadId = threadId;
    this.observer = observer;
    this.fetcher = (input, init) => fetcher(input, init);
    this.connect = connect;
  }

  async start() {
    if (this.disposed) return;
    this.connection?.close();
    this.connection = undefined;
    const revision = ++this.revision;
    await this.refresh();
    if (this.disposed || revision !== this.revision || this.appliedCursor < 0 || this.readOnly)
      return;
    const connection = this.connect(
      `${this.apiUrl}/conversations/${this.threadId}/events?after=${this.cursor}`,
    );
    this.connection = connection;
    connection.addEventListener("update", (event) => {
      if (this.disposed || this.connection !== connection) return;
      try {
        const value = JSON.parse(event.data);
        if (value.conversation_id !== this.threadId || !Number.isSafeInteger(value.seq)) return;
        if (value.seq <= this.cursor) return;
        this.cursor = value.seq;
        clearTimeout(this.timer);
        this.timer = setTimeout(() => void this.refresh(), 75);
      } catch {
        this.observer.error("收到的状态记录无法读取，请重新连接。");
      }
    });
    connection.onerror = () => {
      if (this.disposed || this.connection !== connection) return;
      connection.close();
      this.connection = undefined;
      this.observer.error("状态连接已断开，已有消息会保留。重新连接不会重新执行任务。");
    };
  }

  refresh(): Promise<void> {
    if (this.disposed) return Promise.resolve();
    this.dirty = true;
    if (this.pending) return this.pending;
    this.pending = this.readSnapshots().finally(() => {
      this.pending = undefined;
    });
    return this.pending;
  }

  private async readSnapshots() {
    while (this.dirty && !this.disposed) {
      this.dirty = false;
      const controller = new AbortController();
      this.controller = controller;
      try {
        const snapshot = await requestJson<ConversationSnapshot>(
          `${this.apiUrl}/conversations/${this.threadId}/snapshot`,
          { signal: controller.signal },
          this.fetcher,
        );
        if (this.disposed || controller.signal.aborted) return;
        if (
          snapshot.thread_id !== this.threadId ||
          !Array.isArray(snapshot.messages) ||
          !Number.isSafeInteger(snapshot.event_cursor) ||
          snapshot.messages.some(
            (message) =>
              !message.id ||
              !["human", "ai"].includes(message.type) ||
              !["user", "chat", "progress", "system"].includes(message.source || ""),
          ) ||
          new Set(snapshot.messages.map((message) => message.id)).size !== snapshot.messages.length
        ) {
          throw new Error("会话状态格式不完整，保留已有内容。请重新连接。");
        }
        if (snapshot.event_cursor < this.appliedCursor) continue;
        this.appliedCursor = snapshot.event_cursor;
        this.readOnly = snapshot.read_only;
        this.cursor = Math.max(this.cursor, snapshot.event_cursor);
        this.observer.snapshot(snapshot);
        this.observer.error("");
      } catch (error) {
        if (!this.disposed && !controller.signal.aborted)
          this.observer.error(
            error instanceof Error ? error.message : "无法读取会话状态，请重新连接。",
          );
        return;
      }
    }
  }

  dispose() {
    this.disposed = true;
    ++this.revision;
    this.controller?.abort();
    this.connection?.close();
    this.connection = undefined;
    clearTimeout(this.timer);
  }
}
