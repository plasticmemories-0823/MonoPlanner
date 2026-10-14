import type { Activity } from "./conversation-types";

/** 仅合并用户主动读取的诊断分页，不接入聊天消息或后台订阅。 */
export function appendDiagnosticPage(previous: Activity[], incoming: Activity[]) {
  const rows = new Map(previous.map((row) => [row.id, row]));
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
  }
  return [...rows.values()];
}
