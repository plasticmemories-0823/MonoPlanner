import type { ActivityUsage } from "./activity-usage";

/** 浏览器只接收产品展示协议，不构造模型消息或工具结果。 */
export type Message = {
  id?: string;
  type: "human" | "ai" | "system" | "tool";
  content: string | { type: string; text?: string; [key: string]: unknown }[];
  source?: "user" | "chat" | "progress" | "system" | "work_proposal";
  proposal_status?: string;
};

export type ConversationTitle = {
  title: string;
  status: string;
  revision: number;
};

export type Visibility = "active" | "archived" | "deleted";
export type RuntimeSettings = { model_name: string; reasoning_effort: string; online: boolean };
export type Conversation = {
  thread_id: string;
  created_at: string;
  updated_at: string;
  status: string;
  repo?: string;
  runtime_settings?: RuntimeSettings;
  title?: ConversationTitle;
  visibility?: Visibility;
  read_only?: boolean;
  values?: { messages?: Message[] };
};

export type Activity = {
  id: string;
  task_id?: string;
  source?: "chat" | "work";
  after_id: string;
  kind: "thinking" | "operation";
  label: string;
  detail: string;
  status: string;
  stage?: "trial" | "acceptance";
  summary?: string;
  started_at?: number;
  seconds?: number;
  exit_code?: number;
  child_usage?: { status: string; total_tokens?: number; source?: string };
  append?: boolean;
  version?: number;
  seq?: number;
};

export type UsageKind = "chat" | "work" | "progress" | "title";
export type StorageFailure = {
  code: "storage_unavailable";
  message: string;
  operation: string;
  dispatch_suspended: true;
};
export type AcceptanceLimit = {
  active: boolean;
  completed_rounds: number;
  round_limit: number;
  round_increment: number;
  pause_id: string;
  last_verification_status: string;
};
export type Continuation = {
  protocol: number;
  available: boolean;
  stop_id: string;
  mode: "continue" | "review";
  reason?: string;
  additions: Record<string, number>;
  requirements: string[];
};
export type ConversationSnapshot = {
  thread_id: string;
  status: string;
  chat_status: string;
  work: {
    task_id: string;
    input_id?: string;
    repo?: string;
    status: string;
    reason?: string;
    revision?: number;
    acceptance_limit?: AcceptanceLimit | null;
    continuation?: Continuation | null;
  } | null;
  repo: string;
  runtime_settings?: RuntimeSettings;
  workspace_bound: boolean;
  messages: Message[];
  pending_messages: Message[];
  activities: Activity[];
  usage: Partial<Record<UsageKind, ActivityUsage>>;
  title?: ConversationTitle;
  updated_at?: string;
  event_cursor: number;
  read_only: boolean;
  storage_error?: StorageFailure | null;
};

export type ConversationCommand = {
  id: string;
  kind: "message" | "pause" | "resume" | "supplement" | "workspace" | "confirm_work";
  text?: string;
  repo?: string;
  runtime_settings?: Record<string, unknown>;
  acceptance_pause_id?: string;
  resume_token?: string;
  proposal_id?: string;
};

export type CommandReceipt = {
  received: boolean;
  id: string;
  applied: boolean;
  status: string;
  event_cursor: number;
  recorded?: boolean;
  control_requested?: boolean;
  termination_confirmed?: boolean;
  message?: string;
  storage_error?: StorageFailure | null;
};

export function commandNotice(kind: "pause" | "resume", receipt: CommandReceipt) {
  if (receipt.recorded === false || receipt.received === false)
    return {
      warning: true,
      message: receipt.control_requested
        ? "已请求终止，但记录写入失败；终止结果待核对，不能直接恢复。"
        : "控制请求未确认，记录尚未保存；请核对现场，不会自动重发。",
    };
  return {
    warning: false,
    message: kind === "pause" ? "已发送暂停，正在终止当前操作并保存现场" : "继续请求已保存",
  };
}

export function emptySnapshot(): ConversationSnapshot {
  return {
    thread_id: "",
    status: "idle",
    chat_status: "idle",
    work: null,
    repo: "",
    workspace_bound: false,
    messages: [],
    pending_messages: [],
    activities: [],
    usage: {},
    event_cursor: 0,
    read_only: false,
  };
}

export function isWorking(status?: string): boolean {
  return [
    "queued",
    "running",
    "starting",
    "preparing",
    "needs_plan",
    "executing",
    "pausing",
  ].includes(status || "");
}

/** 与后端控制命令准入一致；停止未确认或输出受限时不能以补充代替恢复。 */
export function canSupplementWork(status?: string): boolean {
  return ["queued", "running", "needs_plan", "executing", "paused", "awaiting_user"].includes(
    status || "",
  );
}

/** 继续先核对原件；普通聊天、补充和重启都不能放行中断任务。 */
export function canResumeWork(work?: ConversationSnapshot["work"]): boolean {
  return Boolean(work?.continuation?.available && work.continuation.stop_id);
}

export function acceptanceLimitLabel(limit?: AcceptanceLimit | null): string | null {
  if (!limit?.active) return null;
  return `已完成 ${limit.completed_rounds} / ${limit.round_limit} 轮验收，等待你决定是否追加 ${limit.round_increment} 轮`;
}

export function resumeCommand(work: ConversationSnapshot["work"], id: string): ConversationCommand {
  return {
    id,
    kind: "resume",
    ...(work?.continuation?.stop_id ? { resume_token: work.continuation.stop_id } : {}),
    ...(work?.acceptance_limit?.active
      ? { acceptance_pause_id: work.acceptance_limit.pause_id }
      : {}),
  };
}
