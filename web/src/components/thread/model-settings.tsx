"use client";

import { useEffect, useState } from "react";
import { Brain, Globe, Settings } from "lucide-react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import { PasswordInput } from "../ui/password-input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "../ui/dialog";

export type RuntimeSettings = {
  model_name: string;
  reasoning_effort: string;
  online: boolean;
};
type Preferences = RuntimeSettings & {
  has_api_key: boolean;
  key_source: string;
};
const effortNames: Record<string, string> = {
  none: "关闭思考",
  low: "low",
  high: "high",
  max: "max",
};
const defaults = {
  model_name: "deepseek-flash",
  reasoning_effort: "high",
  online: false,
  has_api_key: false,
  key_source: "environment",
};

export function ModelSettings({
  apiUrl,
  current,
}: {
  apiUrl: string;
  current?: RuntimeSettings;
}) {
  const [value, setValue] = useState<Preferences>(defaults);
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const [key, setKey] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetch(`${apiUrl}/settings`, { cache: "no-store" })
      .then(async (r) => {
        const result = await r.json();
        if (!r.ok) throw new Error(result.error || "无法读取模型设置");
        if (!cancelled) {
          setValue(result);
          setReady(true);
        }
      })
      .catch((e) => toast.error(String(e)));
    return () => {
      cancelled = true;
    };
  }, [apiUrl]);

  async function save(change: Partial<Preferences> = {}, credential = "") {
    setBusy(true);
    const previous = value;
    const next = { ...value, ...change };
    setValue(next);
    try {
      const r = await fetch(`${apiUrl}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: next.model_name,
          reasoning_effort: next.reasoning_effort,
          online: next.online,
          ...(credential.trim() ? { api_key: credential } : {}),
        }),
      });
      const result = await r.json();
      if (!r.ok) throw new Error(result.error || "保存失败");
      setValue(result);
      setKey("");
      toast.success("已保存；模型、思考和联网设置从下个新任务生效", {
        position: "top-right",
      });
      return true;
    } catch (e) {
      setValue(previous);
      toast.error(String(e));
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function deleteKey() {
    setBusy(true);
    try {
      const r = await fetch(`${apiUrl}/settings/api-key`, { method: "DELETE" });
      const result = await r.json();
      if (!r.ok) throw new Error(result.error || "删除失败");
      setValue(result);
      setKey("");
      toast.success(
        result.has_api_key
          ? "本机保存的密钥已删除，仍可使用环境变量中的密钥"
          : "本机密钥已删除",
        { position: "top-right" },
      );
    } catch (e) {
      toast.error(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div
        className="flex flex-wrap items-center gap-1 text-xs"
        data-testid="model-toolbar"
      >
        <label
          className="text-muted-foreground hover:bg-muted flex h-8 items-center rounded-full px-2"
          title="新任务使用的模型"
        >
          <span className="sr-only">模型</span>
          <select
            aria-label="模型选择"
            value={value.model_name}
            disabled={!ready || busy}
            onChange={(e) => void save({ model_name: e.target.value })}
            className="max-w-36 cursor-pointer rounded-full border-0 bg-transparent py-1 pr-1 text-xs outline-none disabled:opacity-50"
          >
            <option value="deepseek-flash">DeepSeek Flash</option>
          </select>
        </label>
        <label
          className="text-muted-foreground hover:bg-muted flex h-8 items-center gap-1 rounded-full px-2"
          title="新任务的思考程度"
        >
          <Brain className="size-3.5" />
          <span className="sr-only">思考</span>
          <select
            aria-label="思考程度"
            value={value.reasoning_effort}
            disabled={!ready || busy}
            onChange={(e) => void save({ reasoning_effort: e.target.value })}
            className="max-w-24 cursor-pointer rounded-full border-0 bg-transparent py-1 text-xs outline-none disabled:opacity-50"
          >
            {Object.entries(effortNames).map(([id, name]) => (
              <option
                value={id}
                key={id}
              >
                {name}
              </option>
            ))}
          </select>
        </label>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          role="switch"
          aria-label="联网搜索"
          aria-checked={value.online}
          disabled={!ready || busy}
          onClick={() => void save({ online: !value.online })}
          title="新任务的联网开关；网页搜索会产生额外费用"
          className={`h-8 gap-1.5 rounded-full px-2 text-xs ${value.online ? "bg-emerald-50 text-emerald-700 hover:bg-emerald-100" : "text-muted-foreground"}`}
        >
          <Globe className="size-3.5" />
          联网
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => setOpen(true)}
          aria-label="模型设置"
          title={value.has_api_key ? "模型设置" : "设置 · 尚未配置 API Key"}
          className="text-muted-foreground relative size-8 rounded-full p-0"
        >
          <Settings className="size-4" />
          {!value.has_api_key && ready && (
            <span className="absolute top-1.5 right-1.5 size-1.5 rounded-full bg-amber-500" />
          )}
        </Button>
      </div>
      <Dialog
        open={open}
        onOpenChange={(next) => {
          setOpen(next);
          if (!next) setKey("");
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>DeepSeek 设置</DialogTitle>
            <DialogDescription>
              密钥由后端在本机加密保存；留空保存保留现有密钥。
            </DialogDescription>
          </DialogHeader>
          <label
            htmlFor="deepseek-key"
            className="text-sm"
          >
            API Key
          </label>
          <PasswordInput
            id="deepseek-key"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            autoComplete="off"
            placeholder="填写新的 DeepSeek API Key"
            disabled={busy}
          />
          <p
            role="status"
            className="text-muted-foreground text-sm"
          >
            {value.has_api_key
              ? `已配置（${value.key_source === "local" ? "本机加密保存" : "环境变量 / .env"}）`
              : "尚未配置"}
          </p>
          <div className="bg-muted/50 text-muted-foreground rounded-xl p-3 text-xs leading-6">
            <p>
              新任务默认：DeepSeek Flash / {effortNames[value.reasoning_effort]}{" "}
              / 联网{value.online ? "开启" : "关闭"}
            </p>
            {current && (
              <p>
                当前任务：DeepSeek Flash /{" "}
                {effortNames[current.reasoning_effort]} / 联网
                {current.online ? "开启" : "关闭"}
              </p>
            )}
            <p>设置改动从下个新任务生效，当前任务和暂停恢复保持原配置。</p>
          </div>
          <p className="text-muted-foreground text-sm">
            联网使用同一个 Key。网页搜索会增加 DeepSeek API 调用和 Token
            费用；读取网页不额外调用模型。
          </p>
          <div className="flex justify-between gap-3">
            <Button
              type="button"
              variant="outline"
              disabled={busy || value.key_source !== "local"}
              onClick={() => void deleteKey()}
            >
              删除本机密钥
            </Button>
            <Button
              type="button"
              disabled={busy || !ready}
              onClick={async () => {
                if (await save({}, key)) setOpen(false);
              }}
            >
              保存
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
