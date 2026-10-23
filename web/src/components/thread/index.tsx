import { Fragment, ReactNode, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import { BrandMark } from "../brand";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  LoaderCircle,
  PanelRightOpen,
  PanelRightClose,
  SquarePen,
  ArrowUp,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { ResizableHistory } from "./history/resizable";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Workspace, useWorkspace } from "./workspace";
import { CurrentProject } from "./project-execution";
import { useRunControls, statusLabel } from "./run-controls";
import { ActivityRows, useActivity } from "./activity";
import { ReadSources } from "./read-sources";
import { ModelSettings } from "./model-settings";
import { DesktopCards } from "./desktop-controls";
import { DeliveryConfirmation } from "./delivery-confirmation";
import { ExecutionPermissions } from "./execution-permissions";
import { Resources } from "./resources";
import { groupActivity } from "@/lib/work-diagnostics";
import { acceptanceLimitLabel, canResumeWork, canSupplementWork } from "@/lib/conversation-types";

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div
        ref={context.contentRef}
        className={props.contentClassName}
      >
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="h-4 w-4" />
      <span>回到最新消息</span>
    </Button>
  );
}

export function Thread() {
  const [threadId, setThreadId] = useQueryState("threadId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const { repo, setRepo } = useWorkspace(stream.apiUrl, threadId ? stream.values.repo : undefined);
  const { pause, resume, pauseRequested, controlPending } = useRunControls(stream, threadId);
  const messages = stream.messages;
  const [sending, setSending] = useState(false);
  const [sendKind, setSendKind] = useState<"message" | "supplement">("message");
  const acceptsSupplement =
    !stream.values.read_only &&
    !stream.storageError &&
    canSupplementWork(stream.values.work?.status);
  const currentSendKind = acceptsSupplement ? sendKind : "message";
  const pendingCommand = useRef<{
    id: string;
    kind: "message" | "supplement";
    text: string;
    repo: string;
    owner: string | null;
  } | null>(null);
  const isLoading = stream.isLoading;
  const activity = useActivity();
  const activityGroups = groupActivity(messages, activity.items);
  const messageIds = new Set(messages.map((message) => message.id));
  const renderActivity = (items: typeof activity.items) => {
    const tasks = new Map<string, typeof activity.items>();
    for (const row of items.filter((item) => item.source === "work")) {
      const task = row.task_id || "saved-work";
      if (!tasks.has(task)) tasks.set(task, []);
      tasks.get(task)!.push(row);
    }
    return (
      <>
        <ActivityRows
          items={items.filter((row) => row.source === "chat")}
          title="聊天"
        />
        {[...tasks].map(([task, rows]) => (
          <ActivityRows
            key={task}
            items={rows}
            title="工作"
          />
        ))}
      </>
    );
  };

  const lastError = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) {
        // 消息已经记录，保留引用并提前返回。
        return;
      }

      // 消息存在且尚未记录，保存消息并显示错误。
      lastError.current = message;
      toast.error("当前请求未完成，消息和运行记录已保留", {
        description: (
          <p>
            <strong>诊断：</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // 无需操作。
    }
  }, [stream.error]);

  // 收到第一条助手回复时停止等待提示。
  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }

    prevMessageLength.current = messages.length;
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (
      !input.trim() ||
      stream.preparingThread ||
      sending ||
      stream.values.read_only ||
      stream.storageError
    )
      return;
    setFirstTokenReceived(false);
    setSending(true);
    const text = input;
    const kind = currentSendKind;
    if (
      pendingCommand.current?.text !== text ||
      pendingCommand.current.kind !== kind ||
      pendingCommand.current.repo !== repo.trim() ||
      pendingCommand.current.owner !== threadId
    ) {
      pendingCommand.current = {
        id: crypto.randomUUID(),
        kind,
        text,
        repo: repo.trim(),
        owner: threadId,
      };
    }
    try {
      const target = await stream.ensureThread(text, repo.trim());
      pendingCommand.current.owner = target;
      const { owner: _owner, ...command } = pendingCommand.current;
      await stream.command(command, target);
      pendingCommand.current = null;
      setInput((current) => (current === text ? "" : current));
      if (kind === "supplement") toast.success("补充要求已保存，正在中断当前工作并调整剩余步骤");
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "发送结果未确认，输入已保留。请重新连接后核对。",
      );
    } finally {
      setSending(false);
    }
  };

  const chatStarted = !!threadId || !!messages.length;
  return (
    <div className="flex h-screen w-full overflow-hidden">
      <ResizableHistory open={chatHistoryOpen}>
        <ThreadHistory />
      </ResizableHistory>

      <div className="flex min-w-0 flex-1">
        <motion.div
          className={cn(
            "relative flex min-w-0 flex-1 flex-col overflow-hidden",
            !chatStarted && "grid-rows-[1fr]",
          )}
          layout={isLargeScreen}
          transition={
            isLargeScreen ? { type: "spring", stiffness: 300, damping: 30 } : { duration: 0 }
          }
        >
          <header className="relative z-10 flex h-16 shrink-0 items-center justify-between px-3 sm:px-6">
            <div className="flex items-center gap-2">
              <TooltipIconButton
                tooltip="对话记录"
                variant="ghost"
                onClick={() => setChatHistoryOpen((p) => !p)}
              >
                {chatHistoryOpen ? (
                  <PanelRightOpen className="size-5" />
                ) : (
                  <PanelRightClose className="size-5" />
                )}
              </TooltipIconButton>
              <button
                type="button"
                onClick={() => setThreadId(null)}
                className="flex items-center gap-2 rounded-lg px-1"
                aria-label="MonoPlanner 首页"
              >
                <BrandMark className="size-8" />
                <span className="text-base font-semibold tracking-tight">MonoPlanner</span>
              </button>
            </div>
            <TooltipIconButton
              tooltip="新建对话"
              variant="ghost"
              onClick={() => setThreadId(null)}
            >
              <SquarePen className="size-5" />
            </TooltipIconButton>
          </header>

          <StickToBottom className="relative flex-1 overflow-hidden">
            <StickyToBottomContent
              className={cn(
                "absolute inset-0 overflow-y-auto px-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
                !chatStarted && "flex flex-col items-stretch justify-center",
                chatStarted && "grid grid-rows-[1fr_auto]",
              )}
              contentClassName={cn(
                "mx-auto flex w-full max-w-3xl flex-col gap-5",
                chatStarted ? "pt-8 pb-12" : "hidden",
              )}
              content={
                <>
                  {messages
                    .filter((message) => message.type === "human" || message.type === "ai")
                    .map((message) => (
                      <Fragment key={message.id}>
                        {message.type === "human" ? (
                          <HumanMessage
                            key={message.id}
                            message={message}
                            isLoading={isLoading}
                          />
                        ) : (
                          <AssistantMessage
                            key={message.id}
                            message={message}
                            isLoading={isLoading}
                          />
                        )}
                        {message.source === "work_proposal" && (
                          <DeliveryConfirmation message={message} />
                        )}
                        {renderActivity(activityGroups.get(message.id || "") || [])}
                      </Fragment>
                    ))}
                  {renderActivity(
                    [...activityGroups]
                      .filter(([anchor]) => !messageIds.has(anchor))
                      .flatMap(([, items]) => items),
                  )}
                  {stream.values.read_only && (
                    <p className="text-muted-foreground text-sm">
                      当前记录仅供查看，不能在只读状态下派发工作。
                    </p>
                  )}
                  <DesktopCards
                    apiUrl={stream.apiUrl}
                    threadId={threadId}
                    running={isLoading}
                    eventCursor={stream.values.event_cursor}
                    readOnly={stream.values.read_only || Boolean(stream.storageError)}
                  />
                  {stream.storageError && (
                    <p
                      role="alert"
                      className="text-destructive text-sm"
                    >
                      {stream.storageError.message}
                    </p>
                  )}
                  {activity.error && (
                    <p className="text-muted-foreground text-sm">
                      {activity.error}
                      <button
                        type="button"
                        className="ml-2 underline"
                        onClick={activity.reconnect}
                      >
                        重新连接
                      </button>
                    </p>
                  )}
                  {isLoading &&
                    !firstTokenReceived &&
                    !activity.items.some((row) => row.status === "running") && (
                      <AssistantMessageLoading />
                    )}
                </>
              }
              footer={
                <div
                  className={cn(
                    "bg-background sticky bottom-0 flex flex-col items-center pb-4 sm:pb-6",
                    chatStarted ? "pt-3" : "gap-8",
                  )}
                >
                  {!chatStarted && (
                    <div className="mb-2 flex flex-col items-center text-center">
                      <BrandMark className="mb-4 size-20" />
                      <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
                        从一个想法开始
                      </h1>
                      <p className="text-muted-foreground mt-3 text-sm">
                        聊一聊，或选择工作目录后开始构建。
                      </p>
                    </div>
                  )}

                  <ScrollToBottom className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-1/2 mb-4 -translate-x-1/2" />

                  <div
                    className={cn(
                      "bg-background relative z-10 mx-auto w-full max-w-3xl rounded-3xl shadow-[0_4px_28px_-8px_rgba(0,0,0,0.12)] transition-shadow focus-within:shadow-[0_6px_32px_-8px_rgba(0,0,0,0.16)]",
                      "border border-solid",
                    )}
                  >
                    {stream.values.work?.repo && (
                      <CurrentProject
                        key={stream.values.work.task_id}
                        apiUrl={stream.apiUrl}
                        repo={stream.values.work.repo}
                        readOnly={stream.values.read_only || Boolean(stream.storageError)}
                      />
                    )}
                    <form
                      onSubmit={handleSubmit}
                      className="mx-auto grid max-w-3xl grid-rows-[1fr_auto] gap-2"
                    >
                      <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (
                            e.key === "Enter" &&
                            !e.shiftKey &&
                            !e.metaKey &&
                            !e.nativeEvent.isComposing
                          ) {
                            e.preventDefault();
                            const el = e.target as HTMLElement | undefined;
                            const form = el?.closest("form");
                            form?.requestSubmit();
                          }
                        }}
                        aria-label="消息输入框"
                        placeholder="发消息，或描述你想完成的工作…"
                        rows={1}
                        className="placeholder:text-muted-foreground/70 field-sizing-content max-h-48 min-h-12 w-full resize-none border-none bg-transparent px-4 pt-3 pb-1 text-base shadow-none ring-0 outline-none focus:ring-0 focus:outline-none"
                      />

                      <div
                        className="flex items-end justify-between gap-2 px-3 pb-3"
                        data-testid="composer-toolbar"
                      >
                        <div className="flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto whitespace-nowrap">
                          <Workspace
                            apiUrl={stream.apiUrl}
                            repo={repo}
                            onChange={setRepo}
                            disabled={
                              stream.preparingThread ||
                              stream.values.read_only ||
                              Boolean(stream.storageError)
                            }
                            onSelect={stream.selectWorkspace}
                            hasWork={Boolean(stream.values.work)}
                            working={stream.workRunning}
                            previousProject={stream.values.work?.repo}
                          />
                          <ModelSettings
                            apiUrl={stream.apiUrl}
                            current={stream.values.runtime_settings}
                            trailingControls={
                              <ExecutionPermissions
                                apiUrl={stream.apiUrl}
                                threadId={threadId}
                                running={isLoading}
                                ensureThread={stream.ensureThread}
                                eventCursor={stream.values.event_cursor}
                                readOnly={stream.values.read_only || Boolean(stream.storageError)}
                              />
                            }
                            menuItems={
                              <>
                                <ReadSources
                                  apiUrl={stream.apiUrl}
                                  threadId={threadId}
                                  ensureThread={stream.ensureThread}
                                  disabled={
                                    isLoading ||
                                    stream.values.read_only ||
                                    Boolean(stream.storageError)
                                  }
                                />
                                <Resources apiUrl={stream.apiUrl} />
                              </>
                            }
                          />
                        </div>
                        <div className="flex shrink-0 items-center gap-2">
                          {acceptsSupplement && (
                            <select
                              aria-label="消息用途"
                              className="bg-background h-8 max-w-28 rounded-full border px-2 text-xs"
                              value={sendKind}
                              onChange={(e) =>
                                setSendKind(e.target.value as "message" | "supplement")
                              }
                            >
                              <option value="message">聊天</option>
                              <option value="supplement">补充工作要求</option>
                            </select>
                          )}
                          {(isLoading || (stream.storageError && threadId)) && (
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="rounded-full"
                              onClick={pause}
                              disabled={pauseRequested}
                              aria-label={
                                pauseRequested
                                  ? "正在暂停"
                                  : stream.storageError
                                    ? "紧急暂停"
                                    : "暂停"
                              }
                            >
                              <LoaderCircle
                                className={cn("size-4", !stream.storageError && "animate-spin")}
                              />
                              <span className="hidden sm:inline">
                                {pauseRequested
                                  ? "正在暂停"
                                  : stream.storageError
                                    ? "紧急暂停"
                                    : "暂停"}
                              </span>
                            </Button>
                          )}
                          <Button
                            type="submit"
                            size="icon"
                            className="size-9 rounded-full shadow-none"
                            disabled={
                              !input.trim() ||
                              stream.preparingThread ||
                              sending ||
                              stream.values.read_only ||
                              Boolean(stream.storageError)
                            }
                            aria-label={currentSendKind === "supplement" ? "发送补充要求" : "发送"}
                            title={
                              currentSendKind === "supplement"
                                ? "发送补充要求"
                                : "发送聊天消息，不打断当前工作"
                            }
                          >
                            <ArrowUp className="size-5" />
                          </Button>
                        </div>
                      </div>
                    </form>
                    {
                      <p
                        className="text-muted-foreground px-4 pb-2 text-xs"
                        data-testid="usage-summary"
                      >
                        <span title="按已返回缓存用量的输入 Token 加权；含联网辅助调用。">
                          缓存命中：
                          {activity.usage?.cache_rate != null
                            ? `${(activity.usage.cache_rate * 100).toFixed(1)}%${activity.usage.cache_unknown_calls ? "（部分记录）" : ""}`
                            : "暂无数据"}
                        </span>
                        {" · "}
                        <span title="最近一次主模型请求的输入与已完成输出，包含提示词、工具和必要思考续接；等待回复时为估算。不是累计 Token。">
                          上下文（{activity.usage.context_source === "work" ? "编码" : "聊天"}）：
                          {activity.usage?.context && activity.usage.context.window_tokens > 0
                            ? `${activity.usage.context.estimated ? "≈" : ""}${((activity.usage.context.used_tokens / activity.usage.context.window_tokens) * 100).toFixed(1)}%`
                            : "暂无数据"}
                        </span>
                        {activity.usage && (
                          <>
                            {" · "}已记录 Token：
                            {activity.usage.total_tokens.toLocaleString()}
                            {" · "}模型请求：{activity.usage.calls}
                            {activity.usage.unknown_calls > 0 &&
                              ` · ${activity.usage.unknown_calls} 次用量未知`}
                          </>
                        )}
                      </p>
                    }
                    <details
                      className="text-muted-foreground px-4 pb-2 text-xs"
                      data-testid="usage-breakdown"
                    >
                      <summary className="cursor-pointer">用量分类</summary>
                      <div className="mt-2 grid grid-cols-2 gap-2">
                        {(["chat", "work", "progress", "title"] as const).map((kind) => (
                          <span key={kind}>
                            {
                              { chat: "聊天", work: "编码", progress: "进度摘要", title: "标题" }[
                                kind
                              ]
                            }
                            ：{activity.groups[kind]?.total_tokens?.toLocaleString() ?? "未知"}{" "}
                            Token · {activity.groups[kind]?.calls ?? 0} 次
                            {activity.groups[kind]?.unknown_calls ? "（含未知用量）" : ""}
                          </span>
                        ))}
                        {(activity.usage.service_calls ?? 0) > 0 && (
                          <span>
                            联网服务：{activity.usage.service_calls} 次 · 已知费用 ¥
                            {((activity.usage.service_cost_micros ?? 0) / 1000000).toFixed(3)}
                            {(activity.usage.service_unknown_calls ?? 0) > 0 &&
                              ` · ${activity.usage.service_unknown_calls} 次结果待核对，费用预留 ¥${((activity.usage.service_reserved_cost_micros ?? 0) / 1000000).toFixed(3)}`}
                          </span>
                        )}
                      </div>
                    </details>
                    {(isLoading || stream.values.status) && (
                      <div
                        role="status"
                        className="border-border/60 text-muted-foreground flex items-center justify-between gap-3 border-t px-4 py-2 text-xs"
                      >
                        <span className="flex min-w-0 items-center gap-2">
                          <span
                            className={cn(
                              "size-1.5 shrink-0 rounded-full",
                              isLoading ? "animate-pulse bg-emerald-500" : "bg-muted-foreground/50",
                            )}
                          />
                          {stream.storageError
                            ? "记录写入失败 · 已停止派发，执行状态待核对"
                            : isLoading
                              ? stream.workRunning
                                ? `${statusLabel(stream.values.work?.status)} · 聊天不会打断任务`
                                : "正在回复"
                              : (stream.values.work?.status === "acceptance_limit_reached" &&
                                  acceptanceLimitLabel(stream.values.work.acceptance_limit)) ||
                                statusLabel(stream.values.status)}
                        </span>
                        {!stream.workRunning &&
                          canResumeWork(stream.values.work) && (
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="h-6 px-2 text-xs"
                              onClick={resume}
                              disabled={controlPending}
                            >
                              {controlPending ? "正在提交" : stream.values.work?.continuation?.mode === "review" ? "继续复查" : "继续任务"}
                            </Button>
                          )}
                      </div>
                    )}
                  </div>
                </div>
              }
            />
          </StickToBottom>
        </motion.div>
      </div>
    </div>
  );
}
