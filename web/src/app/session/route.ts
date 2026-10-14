import { NextResponse } from "next/server";
import {
  allowedRequest,
  SESSION_COOKIE,
  sessionValue,
} from "@/lib/server-access";

export const dynamic = "force-dynamic";
export async function POST(request: Request) {
  if (
    !allowedRequest(request, true) ||
    Number(request.headers.get("content-length") || 0) > 0
  )
    return NextResponse.json(
      { error: "Local browser request required" },
      { status: 403 },
    );
  try {
    const response = NextResponse.json(
      { ready: true },
      { headers: { "Cache-Control": "no-store" } },
    );
    response.cookies.set(SESSION_COOKIE, sessionValue(), {
      httpOnly: true,
      sameSite: "strict",
      path: "/",
    });
    return response;
  } catch {
    return NextResponse.json(
      { error: "请通过 MonoPlanner 启动器启动服务。" },
      { status: 503 },
    );
  }
}
