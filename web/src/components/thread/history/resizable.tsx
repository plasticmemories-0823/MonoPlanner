import { ReactNode, useEffect, useRef, useState } from "react";

const DEFAULT = 224;
const clamp = (width: number) => Math.min(360, Math.max(180, width));

export function ResizableHistory({
  open,
  children,
}: {
  open: boolean;
  children: ReactNode;
}) {
  const [width, setWidth] = useState(DEFAULT);
  const [ready, setReady] = useState(false);
  const drag = useRef<{ x: number; width: number } | null>(null);
  useEffect(() => {
    const saved = Number(localStorage.getItem("monoplanner.sidebar-width"));
    if (Number.isFinite(saved) && saved > 0) setWidth(clamp(saved));
    setReady(true);
  }, []);
  useEffect(() => {
    if (!ready) return;
    const timer = setTimeout(
      () => localStorage.setItem("monoplanner.sidebar-width", String(width)),
      150,
    );
    return () => clearTimeout(timer);
  }, [width, ready]);
  return (
    <aside
      data-testid="history-sidebar"
      aria-hidden={!open}
      inert={!open}
      className="bg-background relative hidden shrink-0 overflow-hidden border-r lg:block"
      style={{ width: open ? width : 0, borderRightWidth: open ? 1 : 0 }}
    >
      <div
        className="h-full"
        style={{ width }}
      >
        {children}
      </div>
      {open && (
        <div
          role="separator"
          tabIndex={0}
          aria-label="调整对话栏宽度"
          aria-orientation="vertical"
          aria-valuemin={180}
          aria-valuemax={360}
          aria-valuenow={width}
          className="hover:bg-muted focus:bg-muted absolute top-0 right-0 z-30 h-full w-2 cursor-col-resize touch-none"
          onPointerDown={(event) => {
            event.preventDefault();
            drag.current = { x: event.clientX, width };
            event.currentTarget.setPointerCapture(event.pointerId);
          }}
          onPointerMove={(event) => {
            if (drag.current)
              setWidth(
                clamp(drag.current.width + event.clientX - drag.current.x),
              );
          }}
          onPointerUp={(event) => {
            if (drag.current) {
              const final = clamp(
                drag.current.width + event.clientX - drag.current.x,
              );
              setWidth(final);
              localStorage.setItem("monoplanner.sidebar-width", String(final));
            }
            drag.current = null;
          }}
          onPointerCancel={() => {
            drag.current = null;
          }}
          onKeyDown={(event) => {
            if (["ArrowLeft", "ArrowRight", "Home"].includes(event.key)) {
              event.preventDefault();
              setWidth((old) =>
                event.key === "Home"
                  ? DEFAULT
                  : clamp(old + (event.key === "ArrowLeft" ? -16 : 16)),
              );
            }
          }}
        />
      )}
    </aside>
  );
}
