import type { Message } from "@langchain/langgraph-sdk";

/** 切换会话完成渲染后，SDK 才在副作用中重置流和历史记录。 */
export function messagesForThread(
  incoming: Message[],
  selectedKey: string,
  settledKey: string,
  threadId: string | null,
  historyThreadId?: string,
) {
  if (
    !threadId ||
    selectedKey !== settledKey ||
    (historyThreadId && historyThreadId !== threadId)
  )
    return [];
  return incoming;
}

/** 即使收到较旧的恢复快照或空的流视图，也保留已发布消息。 */
export function mergePublicMessages(
  base: Message[],
  incoming: Message[],
): Message[] {
  const result = [...base];
  const positions = new Map(base.map((message, index) => [message.id, index]));
  let changed = false;
  for (const message of incoming) {
    const index = message.id ? positions.get(message.id) : undefined;
    if (index === undefined) {
      if (
        !message.id &&
        result.some((old) => JSON.stringify(old) === JSON.stringify(message))
      )
        continue;
      if (message.id) positions.set(message.id, result.length);
      result.push(message);
      changed = true;
      continue;
    }
    const old = result[index];
    // 界面不支持编辑已发布回复；相同 ID 对应的较短片段
    // 表示较旧快照，不表示删除指令。
    if (
      typeof old.content === "string" &&
      typeof message.content === "string" &&
      message.content.length < old.content.length
    )
      continue;
    if (JSON.stringify(old) !== JSON.stringify(message)) {
      result[index] = message;
      changed = true;
    }
  }
  // 一次更新中的恢复片段和流片段可能先后替换同一 ID。
  // 最终值与归档一致时，不应再次触发 React 渲染。
  return changed && JSON.stringify(result) !== JSON.stringify(base)
    ? result
    : base;
}
