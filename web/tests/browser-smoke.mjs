/** 独立假服务与真实生产界面的浏览器回归；不调用模型或 Docker。 */
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.resolve(process.argv[2] || path.join(root, ".browser-smoke"));
await fs.mkdir(output, { recursive: true });
const conversations = new Map(),
  subscribers = new Map(),
  commands = [],
  requests = [];
const permissions = new Map();
let nextEvent = 0,
  preparing = 0,
  peakSubscriptions = 0,
  failedNextRead = false;
let failNextCreate = true,
  failNextCommand = false;
const stats = { model_calls: 0, docker_calls: 0, commands, requests };
const resourceRequests = [];
const environmentRequests = [];
let projectPinned = true;
const settings = {
  model_name: "deepseek-flash",
  reasoning_effort: "high",
  online: false,
  has_api_key: false,
};
const usage = {
  input_tokens: 100,
  total_tokens: 120,
  output_tokens: 20,
  cached_input_tokens: 80,
  cache_known_input_tokens: 100,
  uncached_input_tokens: 20,
  calls: 1,
  unknown_calls: 0,
  model_seconds: 1,
  cache_unknown_calls: 0,
};
const now = () => new Date().toISOString();
const legacyId = "00000000-0000-4000-8000-000000000001";
conversations.set(legacyId, {
  thread_id: legacyId,
  created_at: "",
  updated_at: "",
  repo: "E:/Fixture/old",
  status: "paused",
  chat_status: "idle",
  work: null,
  event_cursor: 0,
  title: { title: "旧版只读样本", status: "complete", revision: 1 },
  visibility: "active",
  read_only: true,
  messages: [{ id: "legacy-message", type: "ai", content: "旧版只读记录", source: "chat" }],
});
const json = (response, data, code = 200) => {
  response.writeHead(code, { "Content-Type": "application/json", "Cache-Control": "no-store" });
  response.end(JSON.stringify(data));
};
function update(value) {
  value.event_cursor = ++nextEvent;
  value.updated_at = now();
  const event = {
    seq: value.event_cursor,
    kind: "state",
    conversation_id: value.thread_id,
    payload: {},
  };
  for (const response of subscribers.get(value.thread_id) || [])
    response.write(`id: ${event.seq}\nevent: update\ndata: ${JSON.stringify(event)}\n\n`);
}
function snapshot(value) {
  return {
    ...value,
    pending_messages: [],
    activities: [],
    usage: {
      chat: usage,
      work: value.work ? usage : {},
      progress: { ...usage, calls: 0, total_tokens: 0 },
      title: { ...usage, calls: 0, total_tokens: 0 },
    },
    read_only: Boolean(value.read_only),
    workspace_bound: Boolean(value.work),
  };
}
let diagnosticPhase = 1;
const backend = http.createServer(async (request, response) => {
  const url = new URL(request.url, "http://localhost");
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
  requests.push({ method: request.method, path: url.pathname });
  if (url.pathname === "/service") return json(response, { version: "1.0.0", status: "ready" });
  if (url.pathname === "/settings") return json(response, settings);
  if (url.pathname === "/workspace") return json(response, { repo: "" });
  if (url.pathname === "/workspace/directories") {
    const folder = url.searchParams.get("path") || "E:/Fixture/one";
    return json(response, {
      path: folder,
      parent: null,
      directories: [{ name: "项目二", path: "E:/Fixture/two" }],
    });
  }
  if (url.pathname === "/workspace/environment") {
    environmentRequests.push({
      method: request.method,
      repo: body.repo || url.searchParams.get("repo"),
    });
    if (request.method === "POST" && body.action === "prepare") preparing++;
    return json(response, {
      mode: "auto",
      status: "ready",
      stage: "ready",
      message: "已准备，可复用",
      projects: [],
      details: [],
    });
  }
  if (url.pathname === "/resources")
    return json(response, {
      used_bytes: 2 * 1024 ** 3,
      idle_bytes: 1024 ** 3,
      free_bytes: 50 * 1024 ** 3,
      cache_limit_bytes: 20 * 1024 ** 3,
      reserve_bytes: 5 * 1024 ** 3,
      unknown_size: projectPinned,
      resources: projectPinned
        ? [
            {
              id: "environment-fixture",
              owned: true,
              references: 1,
              leases: 0,
              size_bytes: null,
              size_known: false,
            },
          ]
        : [],
      environments: projectPinned
        ? [
            {
              id: "environment-fixture",
              repo: "E:/Fixture/retained",
              size_bytes: null,
              size_known: false,
              pinned: true,
              in_use: false,
              active_task_refs: 0,
            },
          ]
        : [],
    });
  if (url.pathname.startsWith("/resources/")) {
    resourceRequests.push({ path: url.pathname, ...body });
    if (url.pathname === "/resources/project" && body.retain === false) projectPinned = false;
    return json(response, { removed: projectPinned ? [] : ["environment-fixture"], retained: [] });
  }
  if (url.pathname === "/conversations" && request.method === "GET")
    return json(response, { conversations: [...conversations.values()], legacy_read_only: true });
  if (url.pathname === "/conversations" && request.method === "POST") {
    const value = {
      thread_id: body.id,
      created_at: now(),
      updated_at: now(),
      repo: body.repo || "",
      runtime_settings: settings,
      status: "idle",
      chat_status: "idle",
      title: { title: "新对话", status: "temporary", revision: 0 },
      visibility: "active",
      messages: [],
      work: null,
      event_cursor: 0,
    };
    conversations.set(value.thread_id, value);
    if (failNextCreate) {
      failNextCreate = false;
      response.destroy();
      return;
    }
    return json(response, value, 201);
  }
  const match =
    /^\/conversations\/([^/]+)\/(snapshot|events|commands|visibility|purge|diagnostics)$/.exec(
      url.pathname,
    );
  if (match) {
    const value = conversations.get(match[1]);
    if (!value) return json(response, { error: "not found" }, 404);
    if (match[2] === "diagnostics") {
      const after = JSON.parse(url.searchParams.get("after") || "{}");
      const phase = diagnosticPhase;
      const base = { task_id: "work-1", after_id: value.work?.input_id || "", version: phase };
      const items =
        after.work >= phase
          ? []
          : [
              {
                ...base,
                id: "work-1-thinking-1",
                kind: "thinking",
                label: phase === 1 ? "正在思考" : "思考",
                status: phase === 1 ? "running" : "completed",
                detail: phase === 1 ? "工作思考第一段" : "工作思考第一段，已确认要运行检查",
              },
              ...(phase > 1
                ? [
                    {
                      ...base,
                      id: "work-1-operation-1",
                      kind: "operation",
                      label: "正在运行命令",
                      status: value.work?.status === "running" ? "running" : "interrupted",
                      detail: "printf 'offline-operation-marker'",
                    },
                  ]
                : []),
            ];
      return json(response, { items, cursors: { work: phase }, more: false });
    }
    if (match[2] === "snapshot") {
      if (failedNextRead) {
        failedNextRead = false;
        return json(response, { error: "模拟读取失败" }, 503);
      }
      return json(response, snapshot(value));
    }
    if (match[2] === "events") {
      response.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-store" });
      response.write(": connected\n\n");
      const connections = subscribers.get(value.thread_id) || new Set();
      connections.add(response);
      subscribers.set(value.thread_id, connections);
      peakSubscriptions = Math.max(
        peakSubscriptions,
        [...subscribers.values()].reduce((n, entries) => n + entries.size, 0),
      );
      response.on("close", () => connections.delete(response));
      return;
    }
    if (match[2] === "visibility") {
      value.visibility = body.visibility;
      update(value);
      return json(response, value);
    }
    if (match[2] === "purge") {
      conversations.delete(value.thread_id);
      return json(response, { purged: true });
    }
    if (match[2] === "commands") {
      const old = commands.find((command) => command.id === body.id);
      if (!old) {
        commands.push({ ...body, thread_id: value.thread_id });
        if (["message", "supplement"].includes(body.kind)) {
          value.messages.push({ id: body.id, type: "human", content: body.text, source: "user" });
          value.messages.push({
            id: `${body.id}-reply`,
            type: "ai",
            content: `收到：${body.text}`,
            source: "chat",
          });
          value.title = {
            title: `模拟标题：${body.text}`,
            status: "completed",
            revision: value.title.revision + 1,
          };
          if (body.text === "启动假工作") {
            value.work = {
              task_id: "work-1",
              input_id: body.id,
              repo: `${value.repo}/pelican`,
              status: "running",
            };
            preparing++;
            value.status = "running";
          }
        }
        if (body.kind === "pause") {
          if (value.work) value.work.status = "paused";
          value.status = "paused";
        }
        if (body.kind === "resume") {
          value.work.status = "running";
          value.status = "running";
        }
        if (body.kind === "workspace") {
          value.repo = body.repo;
        }
        update(value);
        if (failNextCommand) {
          failNextCommand = false;
          response.destroy();
          return;
        }
      }
      return json(
        response,
        {
          received: true,
          id: body.id,
          applied: true,
          status: value.status,
          event_cursor: value.event_cursor,
        },
        202,
      );
    }
  }
  const legacy = /^\/threads\/([^/]+)\/(permission|read-sources|opens|preview\/[^/]+)$/.exec(
    url.pathname,
  );
  if (legacy) {
    if (legacy[2] === "permission") {
      if (request.method === "POST")
        permissions.set(legacy[1], { requested: body.mode, effective: body.mode });
      return json(
        response,
        permissions.get(legacy[1]) || { requested: "sandbox", effective: "sandbox" },
      );
    }
    if (legacy[2] === "read-sources") return json(response, { sources: [] });
    if (legacy[2] === "opens")
      return json(response, {
        items: [{ id: "preview-1", kind: "preview", path: "demo.html", status: "ready" }],
      });
    if (legacy[2].startsWith("preview/")) {
      response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      response.end("<h1>受控预览</h1><script>window.previewExecuted=true</script>");
      return;
    }
  }
  json(response, { error: `Unsupported fixture route: ${url.pathname}` }, 404);
});
await new Promise((resolve) => backend.listen(0, "127.0.0.1", resolve));
const backendPort = backend.address().port;
const temporary = http.createServer();
await new Promise((resolve) => temporary.listen(0, "127.0.0.1", resolve));
const port = temporary.address().port;
await new Promise((resolve) => temporary.close(resolve));
const child = spawn(
  process.execPath,
  ["node_modules/next/dist/bin/next", "start", "--hostname", "127.0.0.1", "--port", String(port)],
  {
    cwd: root,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      NEXT_TELEMETRY_DISABLED: "1",
      MONOPLANER_UI_PORT: String(port),
      MONOPLANER_BACKEND_PORT: String(backendPort),
      MONOPLANER_SERVICE_TOKEN: "test-only-frontend-fixture-not-a-real-credential",
    },
  },
);
let logs = "";
child.stdout.on("data", (part) => {
  logs += part;
});
child.stderr.on("data", (part) => {
  logs += part;
});
let browser, page;
try {
  const base = `http://127.0.0.1:${port}`;
  for (let attempt = 0; attempt < 100; attempt++) {
    if (child.exitCode !== null) throw new Error(`Frontend exited: ${logs}`);
    try {
      if ((await fetch(base)).ok) break;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  browser = await chromium.launch({ channel: "msedge", headless: true });
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.setDefaultTimeout(10_000);
  page.on("response", (response) => {
    if (response.url().includes("/preview/"))
      stats.preview_response = { status: response.status(), headers: response.headers() };
  });
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto(base);
  await page.getByRole("textbox", { name: "消息输入框" }).waitFor();
  await page.getByRole("button", { name: "执行权限", exact: true }).click();
  await page.getByRole("button", { name: "选择风险询问" }).click();
  await page.getByText("权限未更改", { exact: false }).first().waitFor();
  await page.getByRole("button", { name: "选择风险询问" }).click();
  await page.getByText("当前权限：风险询问", { exact: false }).first().waitFor();
  assert.equal(
    permissions.size,
    1,
    "first-message permission setup creates and binds a conversation",
  );
  assert.equal(
    [...conversations.values()].filter((value) => !value.read_only).length,
    1,
    "retry after an unknown creation result reuses the same conversation ID",
  );
  await page.getByRole("button", { name: "关闭权限选择" }).click();
  const first = new URL(page.url()).searchParams.get("threadId");
  const send = async (text, label = "发送") => {
    await page.getByRole("textbox", { name: "消息输入框" }).fill(text);
    await page.getByRole("button", { name: label, exact: true }).click();
    await page.getByText(`收到：${text}`, { exact: true }).waitFor();
  };
  await page.getByRole("button", { name: "选择工作目录", exact: true }).click();
  assert.equal(await page.getByRole("region", { name: "项目执行环境" }).count(), 0);
  await page.getByRole("button", { name: "选择此目录", exact: true }).click();
  await page.getByRole("button", { name: "关闭目录选择" }).waitFor({ state: "hidden" });
  await send("先讨论项目位置");
  assert.equal(preparing, 0, "selection and discussion do not prepare an environment");
  assert.equal(environmentRequests.length, 0, "browsing does not query the parent environment");
  assert.equal(conversations.get(first).work, null);
  await send("启动假工作");
  assert.equal(preparing, 1, "confirmed work prepares its actual project exactly once");
  const actualProject = conversations.get(first).work.repo;
  assert.equal(actualProject, "E:/Fixture/one/pelican");
  assert.equal(
    conversations.get(first).repo,
    "E:/Fixture/one",
    "workspace authorization remains unchanged",
  );
  const currentProject = page.locator('details[aria-label="当前项目"]');
  await currentProject.getByText(actualProject, { exact: true }).waitFor();
  await currentProject.locator(":scope > summary").click();
  await page
    .getByRole("region", { name: "项目执行环境" })
    .getByText("项目环境已就绪", { exact: true })
    .waitFor();
  assert.ok(environmentRequests.every((request) => request.repo === actualProject));
  assert.equal(preparing, 1, "opening the current project only reads its confirmed environment");
  await currentProject.locator(":scope > summary").click();
  const workActivity = page.locator('[data-testid="activity-group"][data-source="work"]');
  await workActivity
    .locator(":scope > summary")
    .getByText("工作 · 正在思考", { exact: true })
    .waitFor();
  await workActivity.locator(":scope > summary").click();
  await workActivity.locator('[data-activity="thinking"] > summary').click();
  await workActivity.getByText("工作思考第一段", { exact: true }).waitFor();
  const commandsBeforeDisplay = commands.length;
  const preparationsBeforeDisplay = preparing;
  diagnosticPhase = 2;
  await workActivity.getByText("工作思考第一段，已确认要运行检查", { exact: true }).waitFor();
  await workActivity.getByText("printf 'offline-operation-marker'", { exact: true }).waitFor();
  assert.equal(commands.length, commandsBeforeDisplay, "live display never submits a command");
  assert.equal(preparing, preparationsBeforeDisplay, "live display never prepares an environment");
  assert.equal(await workActivity.count(), 1, "incremental fragments retain one activity group");
  assert.equal(await page.getByText("内核原文仅诊断", { exact: true }).count(), 0);
  const before = commands.length;
  await send("现在进度如何");
  assert.deepEqual(
    commands.slice(before).map((command) => command.kind),
    ["message"],
  );
  assert.equal(conversations.get(first).work.status, "running");
  failNextCommand = true;
  await send("确认未知响应");
  await page.waitForFunction(() => !document.querySelector('button[aria-label="发送"]').disabled);
  const knownCommands = commands.length;
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await page.waitForFunction(
    () => document.querySelector('[aria-label="消息输入框"]').value === "",
  );
  assert.equal(commands.length, knownCommands, "explicit retry keeps the saved command ID");
  await page.getByRole("combobox", { name: "消息用途" }).selectOption("supplement");
  await send("保留现有验收", "发送补充要求");
  assert.equal(commands.at(-1).kind, "supplement");
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  await page.getByRole("button", { name: "继续任务", exact: true }).waitFor();
  await page.getByRole("button", { name: "继续任务", exact: true }).click();
  await page.getByRole("button", { name: "暂停", exact: true }).waitFor();
  const beforeReload = commands.length;
  await page.reload();
  await page.getByText("收到：保留现有验收", { exact: true }).waitFor();
  assert.equal(commands.length, beforeReload, "refresh does not replay commands");
  await page.getByRole("button", { name: "选择工作目录", exact: true }).click();
  assert.equal(preparing, 1, "opening directory selection only reads");
  await page.getByRole("button", { name: "项目二", exact: true }).click();
  await page.getByRole("button", { name: "选择此目录", exact: true }).click();
  await page.waitForURL((url) => url.searchParams.get("threadId") !== first);
  await page.waitForFunction(() => document.querySelector('[aria-label="消息输入框"]'));
  assert.equal(preparing, 1, "switching workspace does not prepare its environment");
  assert.equal(await page.locator('details[aria-label="当前项目"]').count(), 0);
  assert.equal(conversations.get(first).work.status, "paused");
  await page.getByRole("button", { name: "预览", exact: true }).click();
  await page
    .frameLocator('iframe[title="隔离文件预览"]')
    .getByRole("heading", { name: "受控预览" })
    .waitFor();
  const previewFrame = page.frames().find((frame) => frame.url().includes("/preview/"));
  assert.equal(await previewFrame.evaluate(() => Boolean(window.previewExecuted)), false);
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "工具与设置", exact: true }).click();
  await page.getByRole("button", { name: "空间与环境", exact: true }).click();
  await page.getByRole("spinbutton", { name: "闲置缓存上限" }).waitFor();
  assert.equal(await page.getByRole("spinbutton", { name: "闲置缓存上限" }).inputValue(), "20");
  assert.equal(await page.getByRole("spinbutton", { name: "数据盘预留" }).inputValue(), "5");
  await page.getByRole("button", { name: "保留或清理", exact: true }).click();
  assert.ok((await page.locator("body").innerText()).includes("待核验"));
  await page.getByRole("button", { name: "保留环境", exact: true }).click();
  await page.getByText("项目环境已保留，后续仍可复用。", { exact: true }).waitFor();
  assert.equal(resourceRequests.at(-1).retain, true);
  await page.getByRole("button", { name: "保留或清理", exact: true }).click();
  await page.getByRole("button", { name: "确认清理", exact: true }).click();
  await page.getByText("尚无已登记的项目环境。", { exact: true }).waitFor();
  assert.deepEqual(resourceRequests.at(-1), {
    path: "/resources/project",
    repo: "E:/Fixture/retained",
    retain: false,
  });
  await page.keyboard.press("Escape");
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "对话记录", exact: true }).click();
  await page
    .getByTestId("history-item")
    .filter({ hasText: "模拟标题：保留现有验收" })
    .getByRole("button")
    .first()
    .click();
  await page.waitForURL((url) => url.searchParams.get("threadId") === first);
  await page.getByText("收到：保留现有验收", { exact: true }).waitFor();
  assert.equal(new URL(page.url()).searchParams.get("threadId"), first);
  assert.equal(await page.getByText("收到：保留现有验收", { exact: true }).count(), 1);
  for (const connection of subscribers.get(first) || []) connection.end();
  await page.getByText("状态连接已断开", { exact: false }).first().waitFor();
  const beforeReconnect = commands.length;
  await page.getByRole("button", { name: "重新连接", exact: true }).first().click();
  await page.waitForFunction(() => !document.body.textContent.includes("状态连接已断开"));
  assert.equal(commands.length, beforeReconnect);
  failedNextRead = true;
  update(conversations.get(first));
  await page.getByText("本地服务暂时不可用", { exact: false }).first().waitFor();
  assert.equal(await page.getByText("收到：保留现有验收", { exact: true }).count(), 1);
  await page.getByRole("button", { name: "重新连接", exact: true }).first().click();
  await page.waitForFunction(
    () => !document.querySelector('[role="status"]')?.textContent.includes("无法连接"),
  );
  assert.equal(commands.length, beforeReconnect);
  await page.screenshot({ path: path.join(output, "workspace.png"), fullPage: true });
  await page
    .getByTestId("history-item")
    .filter({ hasText: "旧版只读样本" })
    .getByRole("button")
    .first()
    .click();
  await page.getByText("旧版只读记录", { exact: true }).waitFor();
  await page.getByRole("textbox", { name: "消息输入框" }).fill("不能发给旧任务");
  assert.equal(await page.getByRole("button", { name: "发送", exact: true }).isDisabled(), true);
  assert.equal(
    await page.getByRole("button", { name: "执行权限", exact: true }).isDisabled(),
    true,
  );
  assert.equal(
    await page.getByRole("button", { name: "选择工作目录", exact: true }).isDisabled(),
    true,
  );
  assert.equal(
    await page.getByRole("button", { name: "管理对话：旧版只读样本", exact: true }).count(),
    0,
  );
  assert.ok(requests.every((request) => request.path !== `/conversations/${legacyId}/events`));
  assert.equal(pageErrors.length, 0, pageErrors.join("\n"));
  assert.ok(
    requests.every(
      (request) =>
        !request.path.includes("/runs") &&
        !request.path.includes("/control") &&
        !request.path.includes("/activity"),
    ),
  );
  // 切换连接的关闭传播允许短暂重叠；每个会话永远不保留重复订阅。
  for (
    let attempt = 0;
    attempt < 20 && [...subscribers.values()].some((connections) => connections.size > 1);
    attempt++
  )
    await new Promise((resolve) => setTimeout(resolve, 50));
  assert.ok([...subscribers.values()].every((connections) => connections.size <= 1));
  stats.result = "passed";
  stats.peak_subscriptions = peakSubscriptions;
  stats.page_errors = pageErrors;
  process.stdout.write("Browser smoke passed: independent fixture, no model or Docker calls.\n");
} catch (error) {
  stats.result = "failed";
  stats.error = String(error.stack || error);
  process.exitCode = 1;
  process.stderr.write(`${stats.error}\n`);
  if (page) {
    await page
      .screenshot({ path: path.join(output, "failure.png"), fullPage: true })
      .catch(() => {});
    stats.page_text = await page
      .locator("body")
      .innerText()
      .catch(() => "");
    stats.frames = await Promise.all(
      page.frames().map(async (frame) => ({
        url: frame.url(),
        text: await frame
          .locator("body")
          .innerText()
          .catch(() => ""),
      })),
    );
  }
} finally {
  await browser?.close();
  child.kill();
  for (const connections of subscribers.values())
    for (const connection of connections) connection.end();
  await new Promise((resolve) => backend.close(resolve));
  await fs.writeFile(path.join(output, "browser-result.json"), JSON.stringify(stats, null, 2));
  await fs.writeFile(path.join(output, "frontend.log"), logs);
}
