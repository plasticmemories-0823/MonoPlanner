import type { Message } from "@/lib/conversation-types";
import { getContentString } from "../utils";
import { CommandBar } from "./shared";
import { memo } from "react";

export const HumanMessage = memo(function HumanMessage({
  message,
  isLoading,
}: {
  message: Message;
  isLoading: boolean;
}) {
  const content = getContentString(message.content);
  return (
    <div className="group ml-auto flex flex-col items-end gap-2">
      <p className="bg-muted w-fit rounded-3xl px-4 py-2 text-right whitespace-pre-wrap">
        {content}
      </p>
      <div className="opacity-0 group-focus-within:opacity-100 group-hover:opacity-100">
        <CommandBar
          content={content}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
});
