import Image from "next/image";
import { cn } from "@/lib/utils";

export function BrandMark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex size-9 shrink-0 items-center justify-center overflow-hidden",
        className,
      )}
    >
      <Image
        src="/brand/monoplanner.png"
        alt="MonoPlanner"
        width={128}
        height={128}
        className="size-full scale-[1.8] object-contain dark:invert"
      />
    </span>
  );
}
