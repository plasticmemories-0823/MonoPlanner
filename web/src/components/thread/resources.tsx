"use client";

import { useEffect, useState } from "react";
import { HardDrive, RefreshCw } from "lucide-react";
import { requestJson } from "@/lib/conversation-api";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "../ui/dialog";

const GiB = 1024 ** 3;
type Resource = {
  id: string;
  kind: string;
  location: string;
  size_bytes: number | null;
  owned: boolean;
  references: number;
  leases: number;
  size_known?: boolean;
  estimated?: boolean;
};
type Environment = {
  id: string;
  repo: string;
  size_bytes: number | null;
  status: string;
  pinned: boolean;
  in_use: boolean;
  size_known?: boolean;
  estimated?: boolean;
  active_task_refs?: number;
};
type Usage = {
  used_bytes: number;
  idle_bytes: number;
  free_bytes: number;
  cache_limit_bytes: number;
  reserve_bytes: number;
  resources: Resource[];
  environments: Environment[];
  unknown_size?: boolean | number;
  unknown_size_count?: number;
  size_known?: boolean;
  estimated?: boolean;
  docker_storage?: {
    status: "known" | "unknown";
    free_bytes?: number | null;
    reserve_bytes?: number;
  };
};
type CleanupResult = { removed?: string[]; retained?: unknown[]; errors?: unknown[] };
const size = (bytes: number | null | undefined, known = true, estimated = false) =>
  !known || bytes == null || !Number.isFinite(bytes)
    ? "待核验"
    : `${estimated ? "≈" : ""}${(bytes / GiB).toFixed(2)} GiB`;

export function Resources({ apiUrl }: { apiUrl: string }) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState<Usage>();
  const [limit, setLimit] = useState("20");
  const [reserve, setReserve] = useState("5");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [revision, refresh] = useState(0);
  const [selected, setSelected] = useState<Environment>();

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setError("");
    void requestJson<Usage>(`${apiUrl}/resources`, { signal: controller.signal })
      .then((next) => {
        if (controller.signal.aborted) return;
        setValue(next);
        setLimit(String(next.cache_limit_bytes / GiB));
        setReserve(String(next.reserve_bytes / GiB));
      })
      .catch((failure) => {
        if (!controller.signal.aborted)
          setError(failure instanceof Error ? failure.message : "无法读取空间信息。");
      });
    return () => controller.abort();
  }, [apiUrl, open, revision]);

  async function save() {
    if (busy) return;
    const cache = Number(limit),
      spare = Number(reserve);
    if (!Number.isFinite(cache) || !Number.isFinite(spare) || cache < 1 || spare < 1) {
      setError("缓存上限和预留空间至少为 1 GiB。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await requestJson(`${apiUrl}/resources/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cache_limit_bytes: Math.floor(cache * GiB),
          reserve_bytes: Math.floor(spare * GiB),
        }),
      });
      setNotice("空间设置已保存。运行中或被任务引用的资料不会因达到缓存上限而删除。");
      refresh((n) => n + 1);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "空间设置未保存。");
    } finally {
      setBusy(false);
    }
  }

  async function cleanup(environment?: Environment, retain = false) {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const releaseProject = environment?.repo && (environment.pinned || retain);
      const result = await requestJson<CleanupResult>(
        `${apiUrl}/resources/${releaseProject ? "project" : "cleanup"}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            releaseProject
              ? { repo: environment.repo, retain }
              : environment
                ? { resource_ids: [environment.id] }
                : {},
          ),
        },
      );
      setNotice(
        retain
          ? "项目环境已保留，后续仍可复用。"
          : `已清理 ${result.removed?.length || 0} 项；保留 ${result.retained?.length || 0} 项${result.errors?.length ? `，${result.errors.length} 项未完成，请核对后再清理` : ""}。`,
      );
      setSelected(undefined);
      refresh((n) => n + 1);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "清理未完成，已有资料保留。");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <button
        type="button"
        className="hover:bg-muted flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm"
        onClick={() => setOpen(true)}
      >
        <HardDrive className="size-4" />
        空间与环境
      </button>
      <Dialog
        open={open}
        onOpenChange={setOpen}
      >
        <DialogContent className="max-h-[calc(100dvh-2rem)] max-w-2xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle>空间与环境</DialogTitle>
            <DialogDescription>
              查看本产品登记的资源。仅清理可回收的闲置资料，不删除正在使用的资源或用户项目。
            </DialogDescription>
          </DialogHeader>
          {value && (
            <div className="grid grid-cols-3 gap-3 text-sm">
              <p>
                {value.unknown_size ? "已知空间（部分）" : "已登记空间"}
                <br />
                <strong>
                  {size(value.used_bytes, value.size_known !== false, value.estimated)}
                  {value.unknown_size ? "（部分未统计）" : ""}
                </strong>
              </p>
              <p>
                可回收缓存
                <br />
                <strong>{size(value.idle_bytes)}</strong>
              </p>
              <p>
                数据盘可用
                <br />
                <strong>{size(value.free_bytes)}</strong>
              </p>
            </div>
          )}
          <p className="text-muted-foreground text-xs">
            Docker 虚拟磁盘占用可能与此不同。此处显示已登记资源的大小，不代表整个硬盘。
          </p>
          {value?.docker_storage && (
            <p className="text-muted-foreground text-xs">
              Docker 数据盘所在磁盘可用：
              {size(value.docker_storage.free_bytes, value.docker_storage.status === "known")} ·
              预留 {size(value.docker_storage.reserve_bytes ?? value.reserve_bytes)}。
              {value.docker_storage.status === "unknown" &&
                "核验完成前不会开始新的环境准备，聊天仍可使用。"}
            </p>
          )}
          <div className="grid grid-cols-2 gap-3">
            <label className="space-y-1 text-sm">
              闲置缓存上限（GiB）
              <Input
                type="number"
                min="1"
                aria-label="闲置缓存上限"
                value={limit}
                onChange={(e) => setLimit(e.target.value)}
              />
            </label>
            <label className="space-y-1 text-sm">
              数据盘预留（GiB）
              <Input
                type="number"
                min="1"
                aria-label="数据盘预留"
                value={reserve}
                onChange={(e) => setReserve(e.target.value)}
              />
            </label>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={busy || !value}
              onClick={() => void save()}
            >
              保存设置
            </Button>
            <Button
              variant="outline"
              disabled={busy || !value}
              onClick={() => void cleanup()}
            >
              回收超额闲置缓存
            </Button>
            <Button
              variant="ghost"
              disabled={busy}
              onClick={() => refresh((n) => n + 1)}
            >
              <RefreshCw className="size-4" />
              刷新
            </Button>
          </div>
          {notice && (
            <p
              role="status"
              className="text-muted-foreground text-sm"
            >
              {notice}
            </p>
          )}
          {error && (
            <p
              role="alert"
              className="text-destructive text-sm"
            >
              {error}
            </p>
          )}
          <section className="space-y-2">
            <h3 className="text-sm font-medium">项目环境</h3>
            {value?.environments.length === 0 && (
              <p className="text-muted-foreground text-sm">尚无已登记的项目环境。</p>
            )}
            {value?.environments.map((environment) => {
              const resource = value.resources.find((row) => row.id === environment.id);
              const blocked =
                environment.in_use ||
                Boolean(environment.active_task_refs) ||
                !resource?.owned ||
                (environment.pinned && !environment.repo);
              return (
                <div
                  key={environment.id}
                  className="flex items-center gap-3 rounded-lg border p-3 text-sm"
                >
                  <div className="min-w-0 flex-1">
                    <p
                      className="truncate"
                      title={environment.repo}
                    >
                      {environment.repo || "共享项目环境"}
                    </p>
                    <p className="text-muted-foreground text-xs">
                      {size(
                        environment.size_bytes,
                        environment.size_known !== false && resource?.size_known !== false,
                        environment.estimated || resource?.estimated,
                      )}{" "}
                      ·{" "}
                      {environment.in_use
                        ? "使用中"
                        : environment.pinned
                          ? "仍被项目或任务引用"
                          : "闲置"}
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={blocked || busy}
                    onClick={() => setSelected(environment)}
                  >
                    {environment.pinned ? "保留或清理" : "清理"}
                  </Button>
                </div>
              );
            })}
          </section>
          {selected && (
            <div className="space-y-3 rounded-lg border p-3">
              <p className="text-sm">
                清理选中的闲置环境后，下次使用可能需要重新下载。项目源码会保留。
                {selected.pinned &&
                  "会取消本项目的环境保留选择；其他任务或项目仍需的资料不会被删除。"}
              </p>
              <div className="flex gap-2">
                <Button
                  disabled={busy}
                  onClick={() => void cleanup(selected)}
                >
                  确认清理
                </Button>
                {selected.pinned && selected.repo && (
                  <Button
                    variant="outline"
                    disabled={busy}
                    onClick={() => void cleanup(selected, true)}
                  >
                    保留环境
                  </Button>
                )}
                <Button
                  variant="ghost"
                  disabled={busy}
                  onClick={() => setSelected(undefined)}
                >
                  取消
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
