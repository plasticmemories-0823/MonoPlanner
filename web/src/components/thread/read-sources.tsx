"use client";

import { useState } from "react";
import { FolderPlus, X } from "lucide-react";
import { Workspace } from "./workspace";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

export function ReadSources({
  apiUrl,
  roots,
  onChange,
  disabled,
}: {
  apiUrl: string;
  roots: string[];
  onChange: (roots: string[]) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        disabled={disabled}
        aria-label="额外可读目录"
        title="管理额外可读目录，仅授权查询资料"
        onClick={() => setOpen(true)}
        className="text-muted-foreground h-8 shrink-0 gap-1 rounded-full px-2 text-xs"
      >
        <FolderPlus className="size-4" />
        资料
        {roots.length > 0 && (
          <span className="text-muted-foreground">{roots.length}</span>
        )}
      </Button>
      <Dialog
        open={open}
        onOpenChange={setOpen}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>额外可读目录</DialogTitle>
            <DialogDescription>
              仅允许查询资料，不授权运行或修改。发送下一条消息时登记至本对话。
            </DialogDescription>
          </DialogHeader>
          <Workspace
            apiUrl={apiUrl}
            repo=""
            disabled={disabled}
            readOnly
            onChange={(path) => onChange([...new Set([...roots, path])])}
          />
          <div className="max-h-64 space-y-2 overflow-y-auto">
            {roots.length === 0 && (
              <p className="text-muted-foreground text-sm">
                尚未添加额外目录。
              </p>
            )}
            {roots.map((path) => (
              <div
                key={path}
                className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm"
              >
                <span className="min-w-0 flex-1 break-all">{path}</span>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  disabled={disabled}
                  aria-label={`移除额外可读目录 ${path}`}
                  onClick={() => onChange(roots.filter((v) => v !== path))}
                >
                  <X className="size-4" />
                </Button>
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
