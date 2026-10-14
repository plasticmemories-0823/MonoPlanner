import { NextRequest, NextResponse } from "next/server";

export function proxy(request: NextRequest) {
  const port = process.env.MONOPLANER_UI_PORT || "3000";
  const host = request.headers.get("host")?.toLowerCase();
  const allowed = ["127.0.0.1", "localhost", "monoplanner.localhost"].map(
    (name) => `${name}:${port}`,
  );
  if (!host || !allowed.includes(host))
    return new NextResponse("Invalid local host", { status: 403 });
  return NextResponse.next();
}
