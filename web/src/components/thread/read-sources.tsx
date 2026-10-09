"use client";

import { X } from "lucide-react";
import { Workspace } from "./workspace";

export function ReadSources({ apiUrl, roots, onChange, disabled }: {
  apiUrl: string; roots: string[]; onChange: (roots: string[]) => void; disabled: boolean;
}) {
  return <div className="flex flex-wrap items-center gap-1" aria-label="授权只读资料">
    <Workspace apiUrl={apiUrl} repo="" disabled={disabled} readOnly
      onChange={(path) => onChange([...new Set([...roots, path])])} />
    {roots.map((path) => <button key={path} type="button" disabled={disabled}
      title={path} aria-label={`移除只读授权 ${path}`} onClick={() => onChange(roots.filter((v) => v !== path))}
      className="text-muted-foreground flex max-w-32 items-center gap-1 rounded-full border px-2 py-1 text-xs">
      <span className="truncate">{path.replace(/[\\/]+$/, "").split(/[\\/]/).pop()}</span><X className="size-3 shrink-0" />
    </button>)}
  </div>;
}
