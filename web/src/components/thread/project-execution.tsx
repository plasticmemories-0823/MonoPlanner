"use client";

import { useEffect, useRef, useState } from "react";
import { CheckCircle2, LoaderCircle, PackageCheck } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { ProjectEnvironmentMonitor, type ProjectEnvironment } from "@/lib/project-environment";

type Image = { id: string; names: string[]; size: string };
const defaultRuntime = { python: "/usr/local/bin/python", conda_env: "" };

export function ProjectExecution({ apiUrl, repo }: { apiUrl: string; repo: string }) {
  return (
    <ProjectEnvironmentCard
      key={`${apiUrl}\0${repo}`}
      apiUrl={apiUrl}
      repo={repo}
    />
  );
}

function ProjectEnvironmentCard({ apiUrl, repo }: { apiUrl: string; repo: string }) {
  const [status, setStatus] = useState<ProjectEnvironment>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [images, setImages] = useState<Image[]>([]);
  const [image, setImage] = useState("");
  const [runtime, setRuntime] = useState(defaultRuntime);
  const [imageError, setImageError] = useState("");
  const [imagesLoading, setImagesLoading] = useState(false);
  const [imageRevision, setImageRevision] = useState(0);
  const monitor = useRef<ProjectEnvironmentMonitor | null>(null);

  useEffect(() => {
    const next = new ProjectEnvironmentMonitor(apiUrl, repo, {
      status(value) {
        setStatus(value);
        if (value.environment) {
          setImage(value.environment.image);
          setRuntime(value.environment.runtime);
        }
      },
      error: setError,
      busy: setBusy,
    });
    monitor.current = next;
    void next.refresh();
    return () => {
      next.dispose();
      monitor.current = null;
    };
  }, [apiUrl, repo]);

  useEffect(() => {
    if (!advanced) return;
    const controller = new AbortController();
    void fetch(`${apiUrl}/execution/images`, {
      signal: controller.signal,
      cache: "no-store",
    })
      .then(async (response) => {
        const value = await response.json();
        if (!response.ok) throw new Error(value.error || "无法读取已有镜像");
        if (!controller.signal.aborted) setImages(value.images || []);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setImageError(String(e.message || e));
      })
      .finally(() => {
        if (!controller.signal.aborted) setImagesLoading(false);
      });
    return () => controller.abort();
  }, [advanced, apiUrl, imageRevision]);

  const preparing = status?.status === "preparing";
  const ready = status?.status === "ready";
  const automatic = status?.mode !== "manual";
  const failed = status?.status === "failed";
  const label = !status
    ? error
      ? "暂时无法读取环境状态"
      : "正在读取环境状态"
    : preparing
      ? "正在准备项目环境"
      : ready
        ? "项目环境已就绪"
        : failed
          ? "项目环境准备未完成"
          : status.status === "cancelled"
            ? "已取消环境准备"
            : "自动准备项目环境";

  return (
    <section
      className="rounded-xl border p-3 text-sm"
      aria-label="项目执行环境"
    >
      <div
        className="flex items-center gap-2 font-medium"
        role="status"
        aria-live="polite"
      >
        {!error && (preparing || !status) ? (
          <LoaderCircle className="size-4 shrink-0 animate-spin" />
        ) : ready ? (
          <CheckCircle2 className="size-4 shrink-0" />
        ) : (
          <PackageCheck className="size-4 shrink-0" />
        )}
        <span>{label}</span>
        {status && (
          <span className="text-muted-foreground ml-auto text-xs font-normal">
            {automatic ? "自动" : "手动"}
          </span>
        )}
      </div>
      <p className="text-muted-foreground mt-2 text-xs leading-5">
        {status?.message ||
          "确认选择目录后自动检查项目，复用已有环境或下载所需工具并安装依赖。浏览目录不会开始安装。"}
      </p>
      {status?.projects && (
        <p className="text-muted-foreground mt-2 text-xs">
          {status.projects.length
            ? `项目：${status.projects.map((project) => `${project.path}（${project.language}）`).join("、")}`
            : "未发现项目依赖，使用基础工具环境。"}
          {status.materials &&
            ` 安装资料：${status.materials.file_count} 个文件，${status.materials.total_bytes.toLocaleString()} 字节。`}
        </p>
      )}
      {status?.error && (
        <p
          role="alert"
          className="text-destructive mt-2 text-xs leading-5"
        >
          {status.error}
        </p>
      )}
      {error && (
        <div
          role="alert"
          className="text-destructive mt-2 flex items-center justify-between gap-2 text-xs"
        >
          <span>{error}</span>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={busy}
            onClick={() => void monitor.current?.refresh()}
          >
            重新连接
          </Button>
        </div>
      )}
      {status && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {preparing ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={busy}
              onClick={() => void monitor.current?.change("cancel")}
            >
              取消准备
            </Button>
          ) : automatic ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={busy}
              onClick={() => void monitor.current?.change("prepare")}
            >
              {busy
                ? "正在处理…"
                : failed || status.status === "cancelled"
                  ? "重新准备"
                  : ready
                    ? "检查并复用环境"
                    : "提前准备"}
            </Button>
          ) : (
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={busy}
              onClick={() => void monitor.current?.change("automatic")}
            >
              {busy ? "正在保存…" : "改为自动准备"}
            </Button>
          )}
          <span className="text-muted-foreground text-xs">已开始的任务继续使用原环境</span>
        </div>
      )}
      {!!status?.details?.length && (
        <details className="mt-3 text-xs">
          <summary className="text-muted-foreground cursor-pointer">准备详情</summary>
          <ul className="mt-2 max-h-28 space-y-1 overflow-y-auto break-words">
            {status.details.map((detail, index) => (
              <li key={index}>{detail}</li>
            ))}
          </ul>
        </details>
      )}
      <details
        className="mt-3 border-t pt-3"
        open={advanced}
        onToggle={(event) => {
          setAdvanced(event.currentTarget.open);
          if (event.currentTarget.open) {
            setImageError("");
            setImagesLoading(true);
          }
        }}
      >
        <summary className="text-muted-foreground cursor-pointer text-xs">
          高级设置 · 使用已有镜像
        </summary>
        {advanced && (
          <div className="mt-3 space-y-3">
            <p className="text-muted-foreground text-xs leading-5">
              已有专用环境时可手动选择；一般项目无需填写。保存后，此项目的新任务使用所选环境。
            </p>
            <label className="block space-y-1">
              <span>本机项目镜像</span>
              <select
                aria-label="Docker 项目镜像"
                value={image}
                disabled={busy || preparing || imagesLoading}
                onChange={(e) => setImage(e.target.value)}
                className="bg-background h-9 w-full rounded-md border px-2 text-sm"
              >
                <option value="">{imagesLoading ? "正在读取已有镜像…" : "请选择已有镜像"}</option>
                {image && !images.some((v) => v.id === image) && (
                  <option value={image}>{image.slice(0, 22)}（尚未在镜像列表找到）</option>
                )}
                {images.map((v) => (
                  <option
                    key={v.id}
                    value={v.id}
                  >
                    {v.names.join(", ") || v.id.slice(0, 22)} · {v.size}
                  </option>
                ))}
              </select>
            </label>
            <label className="block space-y-1">
              <span>容器内 Python 路径</span>
              <Input
                aria-label="容器内 Python 路径"
                value={runtime.python}
                disabled={busy || preparing}
                onChange={(e) => setRuntime({ ...runtime, python: e.target.value })}
              />
            </label>
            <label className="block space-y-1">
              <span>Conda 环境（不用 Conda 时留空）</span>
              <Input
                aria-label="Conda 环境"
                value={runtime.conda_env}
                disabled={busy || preparing}
                onChange={(e) => setRuntime({ ...runtime, conda_env: e.target.value })}
              />
            </label>
            {imageError && (
              <p
                role="alert"
                className="text-destructive text-xs"
              >
                {imageError}
              </p>
            )}
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={busy || preparing || !image}
                onClick={() => void monitor.current?.saveManual(image, runtime)}
              >
                {busy ? "正在保存…" : "保存手动环境"}
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                disabled={imagesLoading}
                onClick={() => {
                  setImageError("");
                  setImagesLoading(true);
                  setImageRevision((value) => value + 1);
                }}
              >
                刷新镜像
              </Button>
            </div>
          </div>
        )}
      </details>
    </section>
  );
}
