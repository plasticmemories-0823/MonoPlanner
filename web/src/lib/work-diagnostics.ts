import type { Activity, ConversationSnapshot, Message } from "./conversation-types";
import { isWorking } from "./conversation-types.ts";

/** 合并只读展示片段；原件覆盖片段，旧分页不能倒退已提交的结束状态。 */
export function appendDiagnosticPage(previous: Activity[], incoming: Activity[]) {
  if (!incoming.length) return previous;
  const rows = new Map(previous.map((row) => [row.id, row]));
  let changed = false;
  for (const row of incoming) {
    const old = rows.get(row.id);
    if (old) {
      if (old.status !== "running" && row.status === "running") continue;
      if (row.version !== undefined && old.version !== undefined && row.version <= old.version)
        continue;
      if (row.seq !== undefined && old.seq !== undefined && row.seq < old.seq) continue;
    }
    rows.set(row.id, {
      ...old,
      ...row,
      detail: row.append ? (old?.detail || "") + row.detail : row.detail,
    });
    changed = true;
  }
  return changed ? [...rows.values()] : previous;
}

export type DiagnosticPage = {
  items: Activity[];
  live?: Activity[];
  cursors: Record<string, number>;
  more: boolean;
  display_warning?: string;
};

/** 一个可见会话只读一条游标链。读取不会提交命令、准备环境或唤醒工作。 */
export class WorkActivityMonitor {
  private cursors: Record<string, number> = {};
  private items: Activity[] = [];
  private controller?: AbortController;
  private timer?: ReturnType<typeof setTimeout>;
  private pending = false;
  private dirty = false;
  private disposed = false;
  private failed = false;
  private active = false;
  private revision = "";
  private warning = "";
  private read: (after: Record<string, number>, signal: AbortSignal) => Promise<DiagnosticPage>;
  private changed: (items: Activity[], error: string) => void;
  private interval: number;

  constructor(
    read: (after: Record<string, number>, signal: AbortSignal) => Promise<DiagnosticPage>,
    changed: (items: Activity[], error: string) => void,
    interval = 400,
  ) {
    this.read = read;
    this.changed = changed;
    this.interval = interval;
  }

  update(revision: string, active: boolean) {
    this.active = active;
    if (this.revision === revision || this.disposed) return;
    this.revision = revision;
    this.refresh();
  }

  reconnect() {
    this.failed = false;
    this.refresh();
  }

  private refresh() {
    if (this.disposed || this.failed) return;
    clearTimeout(this.timer);
    this.dirty = true;
    if (!this.pending) void this.pull();
  }

  private async pull() {
    this.pending = true;
    this.dirty = false;
    const controller = new AbortController();
    this.controller = controller;
    let more = false;
    try {
      const page = await this.read({ ...this.cursors }, controller.signal);
      if (this.disposed || controller.signal.aborted) return;
      const items = appendDiagnosticPage(this.items, [...page.items, ...(page.live || [])]);
      const warning = page.display_warning || "";
      this.cursors = page.cursors;
      more = page.more;
      if (items !== this.items || warning !== this.warning) {
        this.items = items;
        this.warning = warning;
        this.changed(this.items, warning);
      }
    } catch (error) {
      if (this.disposed || controller.signal.aborted) return;
      this.failed = true;
      this.warning =
        error instanceof Error ? error.message : "工作显示暂时中断，请重新连接读取记录。";
      this.changed(this.items, this.warning);
    } finally {
      this.pending = false;
      if (!this.disposed && !this.failed && (this.dirty || more || this.active)) {
        this.timer = setTimeout(() => this.refresh(), this.dirty || more ? 20 : this.interval);
      }
    }
  }

  dispose() {
    this.disposed = true;
    clearTimeout(this.timer);
    this.controller?.abort();
  }
}

const stageLabels: Record<string, string> = {
  queued: "工作已排队",
  starting: "正在检查开工条件",
  preparing: "正在准备项目环境",
  needs_plan: "正在制定工作计划",
  executing: "正在执行工作步骤",
  running: "工作正在进行",
  pausing: "正在停止当前操作",
  paused: "工作已暂停",
  acceptance_limit_reached: "验收轮次已用完，等待手动继续",
  awaiting_user: "工作等待补充或继续",
  preparation_failed: "开工准备未完成",
  task_blocked: "工作需要处理后继续",
  verified_success: "实际验收通过",
  existing_complete: "已有任务已结束",
  control_unconfirmed: "停止结果待核对",
};

/** 即使还没有模型片段，准备阶段也可见；结束状态不能留下永远转动的旧行。 */
export function visibleWorkActivity(
  items: Activity[],
  work: ConversationSnapshot["work"],
): Activity[] {
  if (!work) return items;
  const owns = (row: Activity) =>
    row.task_id === work.task_id || row.id.startsWith(work.task_id + "-");
  // 同一对话开始新任务前，旧任务必须已离开运行态；缺少收尾原件不能冒充成功。
  const visible = items.map((row) =>
    !owns(row) && row.status === "running"
      ? { ...row, status: "unknown", label: "旧任务的执行状态待核对" }
      : row,
  );
  const current = visible.filter(owns);
  if (isWorking(work.status)) {
    if (current.some((row) => row.status === "running")) return visible;
    return [
      ...visible,
      {
        id: work.task_id + "-state",
        task_id: work.task_id,
        after_id: work.input_id || current[0]?.after_id || "",
        kind: "operation",
        label: stageLabels[work.status],
        detail: "",
        status: "running",
      },
    ];
  }
  const ended = ["verified_success", "existing_complete"].includes(work.status)
    ? "completed"
    : [
          "paused",
          "awaiting_user",
          "output_limit_reached",
          "acceptance_limit_reached",
          "task_cancelled",
        ].includes(work.status)
      ? "interrupted"
      : ["preparation_failed", "task_blocked"].includes(work.status)
        ? "failed"
        : "unknown";
  const stopped = visible.map((row) =>
    owns(row) && row.status === "running"
      ? {
          ...row,
          status: ended,
          label: row.kind === "thinking" ? "思考已停止" : row.label.replace(/^正在/, ""),
        }
      : row,
  );
  return [
    ...stopped,
    {
      id: work.task_id + "-state",
      task_id: work.task_id,
      after_id: work.input_id || "",
      kind: "operation",
      label: stageLabels[work.status] || "工作已停止，记录已保留",
      detail: "",
      status: ended,
    },
  ];
}

/** 消息与活动分别保存；这里只计算展示位置，不生成任何模型消息。 */
export function groupActivity(messages: Message[], items: Activity[]) {
  const anchors = new Map<string, string>();
  let user = "";
  for (const message of messages) {
    if (message.type === "human") user = message.id || user;
    if (message.id) anchors.set(message.id, user || message.id);
  }
  const groups = new Map<string, Activity[]>();
  for (const row of items) {
    const anchor = anchors.get(row.after_id) || row.after_id;
    if (!groups.has(anchor)) groups.set(anchor, []);
    groups.get(anchor)!.push(row);
  }
  return groups;
}
