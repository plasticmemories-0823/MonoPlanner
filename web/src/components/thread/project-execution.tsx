"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";

type Image = { id: string; names: string[]; size: string };
const defaultRuntime = { python: "/opt/miniconda3/envs/testbed/bin/python", conda_env: "testbed" };

export function ProjectExecution({ apiUrl, repo }: { apiUrl: string; repo: string }) {
  const [images, setImages] = useState<Image[]>([]);
  const [image, setImage] = useState("");
  const [runtime, setRuntime] = useState(defaultRuntime);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  const currentRepo = useRef(repo);
  currentRepo.current = repo;

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    setSaved(false);
    setImage("");
    setRuntime(defaultRuntime);
    void fetch(`${apiUrl}/workspace/execution?${new URLSearchParams({ repo })}`, { signal: controller.signal, cache: "no-store" })
      .then(async (r) => {
        const value = await r.json();
        if (!r.ok) throw new Error(value.error || "无法读取执行环境");
        if (!controller.signal.aborted) {
          setImage(value.image || "");
          setRuntime(value.runtime || defaultRuntime);
          setSaved(Boolean(value.image));
        }
      }).catch((e) => { if (!controller.signal.aborted) setError(String(e.message || e)); });
    return () => controller.abort();
  }, [apiUrl, repo]);

  useEffect(() => {
    const controller = new AbortController();
    void fetch(`${apiUrl}/execution/images`, { signal: controller.signal, cache: "no-store" })
      .then(async (r) => {
        const value = await r.json();
        if (!r.ok) throw new Error(value.error || "无法读取 Docker 镜像");
        if (!controller.signal.aborted) setImages(value.images);
      }).catch((e) => { if (!controller.signal.aborted) setError(String(e.message || e)); });
    return () => controller.abort();
  }, [apiUrl]);

  async function save() {
    const target = repo;
    setBusy(true);
    setError("");
    setSaved(false);
    try {
      const response = await fetch(`${apiUrl}/workspace/execution`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo: target, image, runtime }),
      });
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || "执行环境保存失败");
      if (currentRepo.current === target) setSaved(true);
    } catch (e) {
      if (currentRepo.current === target) setError(e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); }
  }

  return (
    <details className="rounded-xl border p-3 text-sm">
      <summary className="cursor-pointer font-medium">执行环境 · Docker{saved ? " · 已配置" : image ? " · 待保存" : " · 待配置"}</summary>
      <div className="mt-3 space-y-3">
        <label className="block space-y-1">
          <span>本机项目镜像</span>
          <select aria-label="Docker 项目镜像" value={image} disabled={busy} onChange={(e) => { setImage(e.target.value); setSaved(false); }}
            className="bg-background h-9 w-full rounded-md border px-2 text-sm">
            <option value="">请选择已有镜像</option>
            {image && !images.some((v) => v.id === image) && <option value={image}>{image.slice(0, 22)}（尚未在镜像列表找到）</option>}
            {images.map((v) => <option key={v.id} value={v.id}>{v.names.join(", ") || v.id.slice(0, 22)} · {v.size}</option>)}
          </select>
        </label>
        <label className="block space-y-1"><span>容器内 Python 路径</span>
          <Input aria-label="容器内 Python 路径" value={runtime.python} disabled={busy}
            onChange={(e) => { setRuntime({ ...runtime, python: e.target.value }); setSaved(false); }} />
        </label>
        <label className="block space-y-1"><span>Conda 环境（不用 Conda 时留空）</span>
          <Input aria-label="Conda 环境" value={runtime.conda_env} disabled={busy}
            onChange={(e) => { setRuntime({ ...runtime, conda_env: e.target.value }); setSaved(false); }} />
        </label>
        <p className="text-muted-foreground text-xs leading-5">用于编码、调查、验收和已有脚本运行。只使用已有镜像，不自动下载或安装依赖；已开始执行的任务保留原环境。</p>
        {error && <p role="alert" className="text-destructive text-xs">{error}</p>}
        <div className="flex items-center justify-between gap-2">
          <span role="status" className="text-muted-foreground text-xs">{saved ? "项目环境已保存" : "聊天与资料查询无需 Docker"}</span>
          <Button type="button" variant="outline" size="sm" disabled={busy || !image} onClick={() => void save()}>{busy ? "正在保存…" : "保存执行环境"}</Button>
        </div>
      </div>
    </details>
  );
}
