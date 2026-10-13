export type ProjectRuntime = { python: string; conda_env: string };
export type ProjectBinding = { image: string; runtime: ProjectRuntime };
export type ProjectEnvironment = {
  mode: "auto" | "manual";
  status: "idle" | "preparing" | "ready" | "failed" | "cancelled";
  stage: string;
  message: string;
  updated_at?: string;
  error?: string;
  environment?: ProjectBinding;
  details?: string[];
};

type Observer = {
  status: (value: ProjectEnvironment) => void;
  error: (message: string) => void;
  busy: (value: boolean) => void;
};

/** One in-flight read per selected folder; mutations supersede stale reads. */
export class ProjectEnvironmentMonitor {
  private controller?: AbortController;
  private timer?: ReturnType<typeof setTimeout>;
  private revision = 0;
  private disposed = false;
  private mutating = false;
  private apiUrl: string;
  private repo: string;
  private observer: Observer;
  private fetcher: typeof fetch;
  private interval: number;

  constructor(
    apiUrl: string,
    repo: string,
    observer: Observer,
    fetcher: typeof fetch = fetch,
    interval = 1500,
  ) {
    this.apiUrl = apiUrl;
    this.repo = repo;
    this.observer = observer;
    // Browser fetch must not receive the monitor instance as its receiver.
    this.fetcher = (input, init) => fetcher(input, init);
    this.interval = interval;
  }

  refresh() {
    if (this.mutating) return Promise.resolve();
    return this.request();
  }

  change(action: "prepare" | "automatic" | "cancel") {
    return this.mutate("/workspace/environment", { repo: this.repo, action });
  }

  saveManual(image: string, runtime: ProjectRuntime) {
    return this.mutate("/workspace/execution", {
      repo: this.repo,
      image,
      runtime,
    });
  }

  dispose() {
    this.disposed = true;
    ++this.revision;
    this.controller?.abort();
    clearTimeout(this.timer);
  }

  private async mutate(path: string, body: object) {
    if (this.disposed || this.mutating) return;
    this.mutating = true;
    this.observer.busy(true);
    await this.request(path, body);
    if (!this.disposed) {
      this.mutating = false;
      this.observer.busy(false);
    }
  }

  private async request(path?: string, body?: object) {
    if (this.disposed) return;
    clearTimeout(this.timer);
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    const revision = ++this.revision;
    const current = () => !this.disposed && revision === this.revision;
    this.observer.error("");
    try {
      const response = await this.fetcher(
        path
          ? `${this.apiUrl}${path}`
          : `${this.apiUrl}/workspace/environment?${new URLSearchParams({ repo: this.repo })}`,
        {
          signal: controller.signal,
          cache: "no-store",
          ...(body && {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }),
        },
      );
      const value = await response.json();
      if (!current()) return;
      if (!response.ok)
        throw new Error(value.error || "无法连接环境服务，请稍后重新连接。");
      if (path === "/workspace/execution") {
        // A successful legacy save is followed by the same public status read.
        await this.request();
        return;
      }
      this.observer.status(value);
      if (value.status === "preparing")
        this.timer = setTimeout(() => void this.refresh(), this.interval);
    } catch (error) {
      if (current() && !controller.signal.aborted)
        this.observer.error(
          error instanceof Error ? error.message : String(error),
        );
    }
  }
}
