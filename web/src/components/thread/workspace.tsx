import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ChevronRight,
  Folder,
  FolderOpen,
  LoaderCircle,
} from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

type DirectoryPage = {
  path: string;
  parent: string | null;
  directories: { name: string; path: string }[];
};

export function useWorkspace(apiUrl: string, existing?: string) {
  const [repo, setRepo] = useState("");
  useEffect(() => {
    if (existing !== undefined) {
      setRepo(existing);
      return;
    }
    const saved = localStorage.getItem("monoplaner.repo");
    if (saved) setRepo(saved);
    else
      fetch(`${apiUrl}/workspace`)
        .then((r) => r.json())
        .then((v) => setRepo(v.repo || ""))
        .catch(() => {});
  }, [apiUrl, existing]);
  return { repo, setRepo };
}

export function Workspace({
  apiUrl,
  repo,
  onChange,
  disabled,
  readOnly = false,
}: {
  apiUrl: string;
  repo: string;
  onChange: (value: string) => void;
  disabled: boolean;
  readOnly?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState<DirectoryPage>();
  const [address, setAddress] = useState("");
  const [loading, setLoading] = useState(false);
  const [selecting, setSelecting] = useState(false);
  const [error, setError] = useState("");
  const requestId = useRef(0);

  async function browse(path: string) {
    const id = ++requestId.current;
    setLoading(true);
    setError("");
    try {
      const r = await fetch(
        `${apiUrl}/workspace/directories?${new URLSearchParams({ path })}`,
        { cache: "no-store" },
      );
      const result = await r.json();
      if (!r.ok)
        throw new Error(
          r.status === 404
            ? "目录接口不可用，请在 MonoPlanner 菜单中重启服务。"
            : result.error || result.detail || "无法打开目录",
        );
      if (id === requestId.current) {
        setPage(result);
        setAddress(result.path);
      }
    } catch (e) {
      if (id === requestId.current)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }

  async function select() {
    if (!page?.path || loading || selecting) return;
    setSelecting(true);
    try {
      if (readOnly) {
        onChange(page.path);
        setOpen(false);
        return;
      }
      const r = await fetch(`${apiUrl}/workspace`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo: page.path }),
      });
      const result = await r.json();
      if (!r.ok) throw new Error(result.error || "无法选择目录");
      localStorage.setItem("monoplaner.repo", result.repo);
      onChange(result.repo);
      setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSelecting(false);
    }
  }

  const name =
    repo
      .replace(/[\\/]+$/, "")
      .split(/[\\/]/)
      .pop() || repo;
  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className="text-muted-foreground h-8 max-w-40 gap-1.5 rounded-full px-2.5 text-xs"
        disabled={disabled}
        title={repo || (readOnly ? "添加只读资料目录" : "选择工作目录")}
        aria-label={readOnly ? "添加只读资料目录" : "选择工作目录"}
        onClick={() => {
          setOpen(true);
          void browse(repo);
        }}
      >
        <FolderOpen className="size-4 shrink-0" />
        <span className="truncate">{name || (readOnly ? "只读资料" : "工作目录")}</span>
        <ChevronRight className="size-3 shrink-0" />
      </Button>
      <Dialog
        open={open}
        onOpenChange={(next) => {
          setOpen(next);
          if (!next) ++requestId.current;
        }}
      >
        <DialogContent
          className="max-w-xl"
          closeLabel="关闭目录选择"
        >
          <DialogHeader>
            <DialogTitle className="text-lg font-semibold">
              {readOnly ? "添加只读资料目录" : "选择工作目录"}
            </DialogTitle>
            <DialogDescription className="text-muted-foreground text-sm">
              {readOnly ? "仅授权查询资料，不授权运行或修改。发送下一条消息时登记至本对话。" : "打开文件夹后，选择当前目录。普通聊天无需选择。"}
            </DialogDescription>
          </DialogHeader>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="icon"
              disabled={page?.parent === null || loading}
              aria-label="上一级目录"
              onClick={() => void browse(page?.parent || "")}
            >
              <ArrowLeft className="size-4" />
            </Button>
            <Input
              aria-label="目录路径"
              disabled={loading || selecting}
              value={address}
              placeholder="输入完整路径，或从下方选择"
              onChange={(e) => setAddress(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  e.stopPropagation();
                  void browse(address);
                }
              }}
            />
            <Button
              type="button"
              variant="outline"
              disabled={loading}
              onClick={() => void browse(address)}
            >
              打开
            </Button>
          </div>
          <div
            className="bg-muted/20 h-72 overflow-y-auto rounded-xl border p-2"
            aria-label="文件夹列表"
            aria-busy={loading}
          >
            {loading ? (
              <div className="text-muted-foreground flex h-full items-center justify-center gap-2 text-sm">
                <LoaderCircle className="size-4 animate-spin" />
                正在读取目录
              </div>
            ) : page?.directories.length ? (
              page.directories.map((entry) => (
                <button
                  type="button"
                  key={entry.path}
                  onClick={() => void browse(entry.path)}
                  className="hover:bg-muted focus-visible:outline-ring flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm focus-visible:outline-2"
                >
                  <Folder className="text-muted-foreground size-4 shrink-0" />
                  <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                  <ChevronRight className="text-muted-foreground size-4" />
                </button>
              ))
            ) : error ? (
              <p className="text-muted-foreground p-3 text-sm">
                目录读取失败。请查看下方原因，修复后重新打开。
              </p>
            ) : !page?.path ? (
              <p className="text-muted-foreground p-3 text-sm">
                请选择本机目录，或输入完整路径。
              </p>
            ) : (
              <p className="text-muted-foreground p-3 text-sm">
                这个目录没有子文件夹，可以直接选择。
              </p>
            )}
          </div>
          {error && (
            <p
              role="alert"
              className="text-destructive text-sm"
            >
              {error}
            </p>
          )}
          <div className="flex items-center justify-between gap-3">
            <Button
              type="button"
              variant="ghost"
              onClick={() => void browse("")}
            >
              本机目录
            </Button>
            <Button
              type="button"
              disabled={
                !page?.path ||
                loading ||
                selecting ||
                !!error ||
                address !== page.path
              }
              onClick={() => void select()}
            >
              {selecting ? "正在选择…" : "选择此目录"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
