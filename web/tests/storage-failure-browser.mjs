/** 真实应用服务注入磁盘写入失败；不填满磁盘、不调用模型或 Docker。 */
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = path.dirname(web);
const output = path.resolve(
  process.argv[2] || path.join(root, "validation/storage-failure-browser-20261014-01"),
);
const state = path.join(output, "state"),
  project = path.join(output, "project");
await fs.mkdir(project, { recursive: true });
await fs.mkdir(state, { recursive: true });
const token = "offline-storage-failure-fixture-not-a-real-credential";
const base = "http://127.0.0.1:3372";
const env = {
  ...process.env,
  MONOPLANER_STATE_DIR: state,
  MONOPLANER_SERVICE_TOKEN: token,
  MONOPLANER_BACKEND_PORT: "3072",
  MONOPLANER_UI_PORT: "3372",
  NEXT_TELEMETRY_DISABLED: "1",
  PYTHONUTF8: "1",
  PYTHONDONTWRITEBYTECODE: "1",
};
for (const key of Object.keys(env))
  if (/proxy/i.test(key) || key === "DEEPSEEK_API_KEY") delete env[key];
const children = [],
  logs = {};
function start(name, executable, args, cwd) {
  const child = spawn(executable, args, {
    cwd,
    env,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  logs[name] = "";
  child.stdout.on("data", (data) => {
    logs[name] += data;
  });
  child.stderr.on("data", (data) => {
    logs[name] += data;
  });
  children.push(child);
}
const report = {
  result: "running",
  paid_model_calls: 0,
  docker_calls: 0,
  checks: [],
  errors: [],
  commands: [],
};
async function until(check, timeout = 15000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    if (await check()) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("Timed out waiting for storage-failure browser state");
}
async function api(endpoint) {
  const response = await fetch(`http://127.0.0.1:3072${endpoint}`, {
    headers: { "X-MonoPlanner-Service": token, Origin: base },
  });
  assert.equal(response.status, 200, endpoint);
  return response.json();
}
async function actions() {
  return (await fs.readFile(path.join(state, "fixture-native-actions.jsonl"), "utf8"))
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
}
let browser, page;
try {
  start(
    "backend",
    process.env.MONOPLANER_TEST_PYTHON || "E:/AI/LangGraph/PP_lite/.venv-web/Scripts/python.exe",
    [
      "-B",
      "-m",
      "uvicorn",
      "tests.fixtures.application_server:app",
      "--host",
      "127.0.0.1",
      "--port",
      "3072",
    ],
    root,
  );
  start(
    "frontend",
    process.execPath,
    ["node_modules/next/dist/bin/next", "start", "--hostname", "127.0.0.1", "--port", "3372"],
    web,
  );
  await until(async () => {
    if (children.some((child) => child.exitCode !== null)) throw new Error(JSON.stringify(logs));
    try {
      return (await api("/service")).version && (await fetch(base)).ok;
    } catch {
      return false;
    }
  });
  browser = await chromium.launch({ channel: "msedge", headless: true });
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.setDefaultTimeout(15000);
  page.on("pageerror", (error) => report.errors.push(error.message));
  page.on("request", (request) => {
    if (request.method() === "POST" && new URL(request.url()).pathname.endsWith("/commands"))
      report.commands.push(request.postDataJSON());
  });
  await page.goto(base);
  const environmentRequests = [];
  let environmentState = "ready";
  const environmentRoute = async (route) => {
    const request = route.request();
    environmentRequests.push({ method: request.method(), url: request.url() });
    if (request.method() === "POST") {
      environmentState = "failed";
      return route.fulfill({ status: 503, contentType: "application/json",
        body: JSON.stringify({ error: "离线模拟：暂时无法核验资源状态" }) });
    }
    return route.fulfill({ contentType: "application/json", body: JSON.stringify({
      mode: "auto", status: environmentState, stage: environmentState,
      message: environmentState === "ready" ? "项目环境已就绪" : "镜像已不存在，请明确重新准备",
    }) });
  };
  await page.route("**/api/workspace/environment*", environmentRoute);
  await page.getByRole("button", { name: "选择工作目录", exact: true }).click();
  await page.getByRole("textbox", { name: "目录路径" }).fill(project);
  await page.getByRole("button", { name: "打开", exact: true }).click();
  const environmentCard = page.getByRole("region", { name: "项目执行环境" });
  await environmentCard.getByRole("button", { name: "检查并复用环境" }).click();
  await environmentCard.getByText("环境状态待核对", { exact: true }).waitFor();
  const beforeReconnect = environmentRequests.filter((item) => item.method === "POST").length;
  await environmentCard.getByRole("button", { name: "重新连接", exact: true }).click();
  await environmentCard.getByText("镜像已不存在，请明确重新准备", { exact: true }).waitFor();
  assert.equal(environmentRequests.filter((item) => item.method === "POST").length, beforeReconnect);
  assert.equal(new URL(environmentRequests.at(-1).url).searchParams.get("verify"), "1");
  assert.equal(await environmentCard.getByText("项目环境已就绪", { exact: true }).count(), 0);
  await page.screenshot({ path: path.join(output, "resource-state-reconnected.png"), fullPage: true });
  report.checks.push("resource failure removes stale ready display; reconnect verifies without reinstall or resend");
  await page.unroute("**/api/workspace/environment*", environmentRoute);
  await page.getByRole("button", { name: "选择此目录", exact: true }).click();
  await page.getByRole("button", { name: "关闭目录选择" }).waitFor({ state: "hidden" });
  const send = async (text) => {
    await page.getByRole("textbox", { name: "消息输入框" }).fill(text);
    await page.getByRole("button", { name: "发送", exact: true }).click();
    await page.waitForFunction(
      () => document.querySelector('[aria-label="消息输入框"]').value === "",
    );
  };
  await send("准备失败工作");
  await page.getByText("开工准备失败", { exact: true }).waitFor();
  await page.getByText("开工准备失败：离线模拟磁盘空间不足", { exact: false }).first().waitFor();
  assert.equal(await page.getByRole("button", { name: "暂停", exact: true }).count(), 0);
  await send("你好");
  await page.getByText("已收到：你好。这次聊天不改变后台工作。", { exact: true }).waitFor();
  const beforeReload = report.commands.length;
  await page.reload();
  await page.getByText("开工准备失败", { exact: true }).waitFor();
  assert.equal(report.commands.length, beforeReload);
  assert.equal((await actions()).filter((item) => item.kind === "work_started").length, 1);
  report.checks.push(
    "preparation failure is immediately visible; chat still works and refresh does not retry work",
  );

  await page.getByRole("button", { name: "新建对话", exact: true }).click();
  await send("开始工作");
  await page.getByRole("button", { name: "暂停", exact: true }).waitFor();
  const thread = new URL(page.url()).searchParams.get("threadId");
  report.thread_id = thread;
  await fs.writeFile(path.join(state, "fixture-storage-failure"), "inject write failure only\n");
  await page.getByRole("alert").filter({ hasText: "本地记录写入失败" }).first().waitFor();
  assert.equal(await page.locator('[role="status"]').filter({ hasText: "聊天不会打断任务" }).count(), 0);
  await page.getByRole("textbox", { name: "消息输入框" }).fill("不能继续派发");
  assert.equal(await page.getByRole("button", { name: "发送", exact: true }).isDisabled(), true);
  assert.equal(await page.getByRole("button", { name: "继续任务", exact: true }).count(), 0);
  report.checks.push(
    "unpersisted SSE health stops new dispatch display and preserves existing messages",
  );

  const blocked = await page.evaluate(async (id) => {
    const response = await fetch(`/api/conversations/${id}/commands`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: crypto.randomUUID(),
        kind: "message",
        text: "blocked fixture request",
      }),
    });
    return { status: response.status, data: await response.json() };
  }, thread);
  assert.equal(blocked.status, 503);
  assert.equal(blocked.data.storage_error.code, "storage_unavailable");
  report.checks.push(
    "same-origin proxy preserves the public storage health code on a rejected command",
  );
  const pauseResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname.endsWith("/commands"),
  );
  await page.getByRole("button", { name: "紧急暂停", exact: true }).click();
  const pause = await (await pauseResponse).json();
  assert.equal(pause.recorded, false);
  assert.equal(pause.control_requested, true);
  assert.equal(pause.termination_confirmed, false);
  await page
    .getByText("已请求终止，但记录写入失败；终止结果待核对，不能直接恢复。", { exact: true })
    .waitFor();
  await until(async () => (await actions()).some((item) => item.kind === "pause_signalled"));
  report.checks.push(
    "emergency pause still reaches cancellation when recording fails and never claims stopped or saved",
  );
  await page.screenshot({ path: path.join(output, "storage-failure.png"), fullPage: true });
  const afterPause = report.commands.length;
  await page.reload();
  await page.getByRole("button", { name: "紧急暂停", exact: true }).waitFor();
  await page.getByText("开始工作", { exact: true }).first().waitFor();
  assert.equal(report.commands.length, afterPause);
  const native = await actions();
  assert.equal(native.filter((item) => item.kind === "work_started").length, 2);
  assert.equal(native.filter((item) => item.kind === "pause_signalled").length, 1);
  report.actions = native;
  report.checks.push("refresh retains the storage fault without replaying work or emergency pause");
  assert.equal(report.errors.length, 0);
  report.result = "passed";
  process.stdout.write("Storage failure browser integration passed.\n");
} catch (error) {
  report.result = "failed";
  report.error = String(error.stack || error);
  process.exitCode = 1;
  process.stderr.write(`${report.error}\n`);
  if (page) {
    await page
      .screenshot({ path: path.join(output, "failure.png"), fullPage: true })
      .catch(() => {});
    report.page_text = await page
      .locator("body")
      .innerText()
      .catch(() => "");
  }
} finally {
  await browser?.close();
  for (const child of children) child.kill();
  await Promise.all(
    Object.entries(logs).map(([name, value]) =>
      fs.writeFile(path.join(output, `${name}.log`), value),
    ),
  );
  await fs.writeFile(path.join(output, "browser-result.json"), JSON.stringify(report, null, 2));
}
