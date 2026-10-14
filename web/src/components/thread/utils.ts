import type { Message } from "@/lib/conversation-types";

/** 从消息内容提取文本摘要：字符串原样返回，多模态内容仅合并文本片段。 */
export function getContentString(content: Message["content"]): string {
  if (typeof content === "string") return content;
  const texts = content
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map((c) => c.text);
  return texts.join(" ");
}
