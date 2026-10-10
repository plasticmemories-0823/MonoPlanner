import type { Message } from "@langchain/langgraph-sdk";

/** Public replies remain visible; the turn's private activity shares one group. */
export function activityAnchors(messages: Message[]) {
  const anchors = new Map<string, string>();
  let current = "";
  for (const message of messages) {
    if (message.type === "human") current = message.id || "";
    if (message.id) anchors.set(message.id, current);
  }
  return anchors;
}
