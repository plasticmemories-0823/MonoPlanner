export type ProjectRuntime = { python: string; conda_env: string };
export type ProjectBinding = { image: string; runtime: ProjectRuntime };
export type ProjectEnvironment = {
  mode: "auto" | "manual";
  status: "idle" | "preparing" | "ready" | "failed" | "cancelled" | "interrupted";
  stage: string;
  message: string;
  updated_at?: string;
  error?: string;
  environment?: ProjectBinding;
  details?: string[];
  projects?: { path: string; language: string; manager: string }[];
  materials?: { file_count: number; total_bytes: number } | null;
};

type Observer = {
  status: (value: ProjectEnvironment) => void;
  error: (message: string) => void;
  busy: (value: boolean) => void;
};

/** 每个选中目录最多保留一个进行中的读取请求；修改操作优先于过期读取。 */
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
  private timeout: number;

  constructor(
    apiUrl: string,
    repo: string,
    observer: Observer,
    fetcher: typeof fetch = fetch,
    interval = 1500,
    timeout = 10000,
  ) {
    this.apiUrl = apiUrl;
    this.repo = repo;
    this.observer = observer;
    // 调用浏览器 fetch 时，不能将监视器实例绑定为其接收者。
    this.fetcher = (input, init) => fetcher(input, init);
    this.interval = interval;
    this.timeout = timeout;
  }

  refresh(verify = false) {
    if (this.mutating) return Promise.resolve();
    return this.request(undefined, undefined, verify);
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

  private async request(path?: string, body?: object, verify = false) {
    if (this.disposed) return;
    clearTimeout(this.timer);
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    const revision = ++this.revision;
    const current = () => !this.disposed && revision === this.revision;
    let timedOut = false;
    const deadline = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.timeout);
    this.observer.error("");
    try {
      const query = new URLSearchParams({ repo: this.repo });
      if (verify) query.set("verify", "1");
      const response = await this.fetcher(
        path
          ? `${this.apiUrl}${path}`
          : `${this.apiUrl}/workspace/environment?${query}`,
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
        // 手动环境保存成功后，读取包含 mode/status 的统一公开状态。
        await this.request();
        return;
      }
      this.observer.status(value);
      if (value.status === "preparing")
        this.timer = setTimeout(() => void this.refresh(), this.interval);
    } catch (error) {
      if (current() && (timedOut || !controller.signal.aborted))
        this.observer.error(
          timedOut
            ? "环境请求超时，结果尚未确认。请重新连接读取状态；不会自动重新准备。"
            : error instanceof Error ? error.message : String(error),
        );
    } finally {
      clearTimeout(deadline);
    }
  }
}
