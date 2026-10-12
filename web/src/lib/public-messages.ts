import type { Message } from "@langchain/langgraph-sdk";

/** The SDK resets stream/history in effects, after a thread selection renders. */
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

/** Published messages survive older recovery snapshots and empty stream views. */
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
    // This UI does not support editing published replies. A shorter fragment
    // with the same ID is an older snapshot, not a deletion instruction.
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
  // Recovery and stream fragments can replace the same ID twice in one update.
  // A final value equal to the archive must not trigger another React render.
  return changed && JSON.stringify(result) !== JSON.stringify(base)
    ? result
    : base;
}
