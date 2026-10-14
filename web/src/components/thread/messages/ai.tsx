import type { Message } from "@/lib/conversation-types";
import { getContentString } from "../utils";
import { CommandBar } from "./shared";
import { MarkdownText } from "../markdown-text";
import { memo } from "react";

// 公开的图消息不包含工具参数或私有诊断信息。
export const AssistantMessage = memo(function AssistantMessage({
  message,
  isLoading,
}: {
  message: Message | undefined;
  isLoading: boolean;
}) {
  if (message?.type !== "ai") return null;
  const content = getContentString(message.content);
  return (
    <div className="group mr-auto flex w-full flex-col gap-2">
      <div className="py-1">
        <MarkdownText>{content}</MarkdownText>
      </div>
      <div className="opacity-0 group-focus-within:opacity-100 group-hover:opacity-100">
        <CommandBar
          content={content}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
});

export function AssistantMessageLoading() {
  return (
    <div className="mr-auto flex items-start gap-2">
      <div className="bg-muted flex h-8 items-center gap-1 rounded-2xl px-4 py-2">
        <div className="bg-foreground/50 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_infinite] rounded-full"></div>
        <div className="bg-foreground/50 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_0.5s_infinite] rounded-full"></div>
        <div className="bg-foreground/50 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_1s_infinite] rounded-full"></div>
      </div>
    </div>
  );
}
