"use client";

import { useRef, useState } from "react";
import { useStreamContext } from "@/providers/Stream";
import type { Message } from "@/lib/conversation-types";
import { Button } from "../ui/button";

export function DeliveryConfirmation({ message }: { message: Message }) {
  const stream = useStreamContext();
  const commandId = useRef<string | null>(null);
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");
  const proposalId = message.id?.replace(/^work-proposal:/, "") || "";
  const recorded = ["pending", "running", "applied", "unknown"].includes(
    message.proposal_status || "",
  );
  const disabled = sending || sent || recorded || stream.values.read_only || Boolean(stream.storageError);

  async function confirm() {
    if (disabled || !/^[a-f0-9]{64}$/.test(proposalId)) return;
    commandId.current ||= crypto.randomUUID();
    setSending(true);
    setError("");
    try {
      await stream.command({
        id: commandId.current,
        kind: "confirm_work",
        proposal_id: proposalId,
      });
      setSent(true);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "确认未完成，请核对记录后重试。");
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="flex flex-col items-start gap-2">
      <Button disabled={disabled} onClick={confirm}>
        {sent || recorded ? "确认已记录" : sending ? "正在确认…" : "确认此方案并执行"}
      </Button>
      {error && <p role="alert" className="text-destructive text-sm">{error}</p>}
    </div>
  );
}
