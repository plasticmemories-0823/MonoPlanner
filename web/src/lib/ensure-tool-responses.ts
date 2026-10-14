import { v4 as uuidv4 } from "uuid";
import { Message, ToolMessage } from "@langchain/langgraph-sdk";

export const DO_NOT_RENDER_ID_PREFIX = "do-not-render-";

export function ensureToolCallsHaveResponses(messages: Message[]): Message[] {
  const newMessages: ToolMessage[] = [];

  messages.forEach((message, index) => {
    if (message.type !== "ai" || message.tool_calls?.length === 0) {
      // 非 AI 消息或没有工具调用时直接跳过。
      return;
    }
    // 存在工具调用时，检查紧随其后的消息是否为工具消息。
    const followingMessage = messages[index + 1];
    if (followingMessage && followingMessage.type === "tool") {
      // 后续已有工具消息，无需补充。
      return;
    }

    // 后续不是工具消息，需要补充新的工具响应消息。
    newMessages.push(
      ...(message.tool_calls?.map((tc) => ({
        type: "tool" as const,
        tool_call_id: tc.id ?? "",
        id: `${DO_NOT_RENDER_ID_PREFIX}${uuidv4()}`,
        name: tc.name,
        content: "Successfully handled tool call.",
      })) ?? []),
    );
  });

  return newMessages;
}
