"use client";

import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import { ThreadProvider } from "@/providers/Thread";
import { Toaster } from "@/components/ui/sonner";
import { LocalSession } from "@/providers/LocalSession";
import React from "react";

export default function HomePage(): React.ReactNode {
  return (
    <React.Suspense fallback={<div>正在加载 MonoPlanner…</div>}>
      <Toaster />
      <LocalSession>
        <ThreadProvider>
          <StreamProvider>
            <Thread />
          </StreamProvider>
        </ThreadProvider>
      </LocalSession>
    </React.Suspense>
  );
}
