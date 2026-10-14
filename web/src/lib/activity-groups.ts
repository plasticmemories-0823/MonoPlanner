import type { Message } from "@langchain/langgraph-sdk";

/** 保留公开回复的显示，同一轮的私有活动归入一组。 */
export function activityAnchors(messages: Message[]) {
  const anchors = new Map<string, string>();
  let current = "";
  for (const message of messages) {
    if (message.type === "human") current = message.id || "";
    if (message.id) anchors.set(message.id, current);
  }
  return anchors;
}
