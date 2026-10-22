/** 真实应用调度、SQLite、安全路由与生产前端；模型、工作和安装由离线夹具替代。 */
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = path.dirname(web);
const output = path.resolve(
  process.argv[2] || path.join(root, "validation/integration-browser-20261014-01"),
);
const state = path.join(output, "state");
const project = path.join(output, "project-a"),
  anotherProject = path.join(output, "project-b");
const readFile = path.join(output, "authorized-note.txt");
await fs.mkdir(project, { recursive: true });
await fs.mkdir(anotherProject, { recursive: true });
await fs.mkdir(state, { recursive: true });
await fs.writeFile(path.join(project, "demo.py"), 'print("offline-fixture-marker")\n');
await fs.writeFile(readFile, "authorized-single-file-marker\n");
await fs.writeFile(path.join(output, "unselected-note.txt"), "not-authorized-sibling-marker\n");
await fs.writeFile(
  path.join(project, "demo.html"),
  '<!doctype html><meta charset="utf-8"><h1>集成预览</h1><script>window.previewExecuted=true</script>',
);
const token = "offline-integration-fixture-not-a-real-credential";
const env = {
  ...process.env,
  MONOPLANER_STATE_DIR: state,
  MONOPLANER_SERVICE_TOKEN: token,
  MONOPLANER_BACKEND_PORT: "3071",
  MONOPLANER_UI_PORT: "3371",
  MONOPLANER_FIXTURE_READ_FILE: readFile,
  NEXT_TELEMETRY_DISABLED: "1",
  PYTHONUTF8: "1",
};
for (const key of Object.keys(env))
  if (/proxy/i.test(key) || key === "DEEPSEEK_API_KEY") delete env[key];
const python =
  process.env.MONOPLANER_TEST_PYTHON || path.join(root, ".venv-web/Scripts/python.exe");
const processes = [],
  logs = {};
function start(name, executable, args, cwd) {
  const child = spawn(executable, args, {
    cwd,
    env,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  logs[name] = "";
  child.stdout.on("data", (value) => {
    logs[name] += value;
  });
  child.stderr.on("data", (value) => {
    logs[name] += value;
  });
  processes.push(child);
  return child;
}
const backend = start(
  "backend",
  python,
  [
    "-m",
    "uvicorn",
    "tests.fixtures.application_server:app",
    "--host",
    "127.0.0.1",
    "--port",
    "3071",
  ],
  root,
);
const frontend = start(
  "frontend",
  process.execPath,
  ["node_modules/next/dist/bin/next", "start", "--hostname", "127.0.0.1", "--port", "3371"],
  web,
);
const base = "http://127.0.0.1:3371";
async function api(endpoint) {
  const response = await fetch(`http://127.0.0.1:3071${endpoint}`, {
    headers: { "X-MonoPlanner-Service": token, Origin: base },
  });
  if (!response.ok) throw new Error(`${endpoint}: ${response.status} ${await response.text()}`);
  return response.json();
}
async function until(check, timeout = 15_000) {
  const end = Date.now() + timeout;
  let last;
  while (Date.now() < end) {
    try {
      const value = await check();
      if (value) return value;
    } catch (error) {
      last = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw last || new Error("Timed out waiting for integration state");
}
let browser, page;
const report = {
  paid_model_calls: 0,
  docker_calls: 0,
  requests: [],
  errors: [],
  checks: [],
  snapshots: [],
};
try {
  await until(async () => {
    if (backend.exitCode !== null || frontend.exitCode !== null)
      throw new Error(JSON.stringify(logs));
    return (await api("/service")).version && (await fetch(base)).ok;
  });
  browser = await chromium.launch({ channel: "msedge", headless: true });
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.setDefaultTimeout(15_000);
  page.on("pageerror", (error) => report.errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error" && /same key|unique .key./i.test(message.text()))
      report.errors.push(message.text());
  });
  page.on("request", (request) => {
    if (request.url().includes("/api/"))
      report.requests.push({ method: request.method(), path: new URL(request.url()).pathname });
  });
  page.on("response", async (response) => {
    if (new URL(response.url()).pathname.endsWith("/snapshot") && response.ok()) {
      const value = await response.json().catch(() => null);
      if (value)
        report.snapshots.push({
          thread_id: value.thread_id,
          event_cursor: value.event_cursor,
          messages: value.messages?.length,
          activities: value.activities?.length,
        });
    }
  });
  await page.goto(base);
  await page.getByRole("textbox", { name: "消息输入框" }).waitFor();
  await page.getByRole("button", { name: "执行权限", exact: true }).click();
  await page.getByRole("button", { name: "选择风险询问" }).click();
  await until(
    async () =>
      (await page.getByRole("button", { name: "选择风险询问" }).getAttribute("aria-pressed")) ===
      "true",
  );
  await page.getByRole("button", { name: "关闭权限选择" }).click();
  const first = new URL(page.url()).searchParams.get("threadId");
  assert.ok(first);
  report.first_thread = first;
  report.checks.push("first-message permission creates one conversation");
  async function chooseProject(repo) {
    await page.getByRole("button", { name: "选择工作目录", exact: true }).click();
    await page.getByRole("textbox", { name: "目录路径" }).fill(repo);
    await page.getByRole("button", { name: "打开", exact: true }).click();
    await page.getByRole("button", { name: "选择此目录", exact: true }).click();
    await page.getByRole("button", { name: "关闭目录选择" }).waitFor({ state: "hidden" });
  }
  await chooseProject(project);
  await until(async () => (await api(`/conversations/${first}/snapshot`)).repo === project);
  assert.equal(await page.getByRole("region", { name: "项目执行环境" }).count(), 0);
  assert.ok(report.requests.every((request) => !request.path.endsWith("/workspace/environment")));
  report.checks.push("workspace selection does not inspect or prepare the parent environment");
  await page.getByRole("button", { name: "工具与设置", exact: true }).click();
  await page.getByRole("button", { name: "额外可读资料", exact: true }).click();
  await page.getByRole("button", { name: "选择文件", exact: true }).click();
  await page.getByText(readFile, { exact: true }).waitFor();
  assert.deepEqual((await api(`/threads/${first}/read-sources`)).sources, [
    { path: readFile, kind: "file" },
  ]);
  await page.keyboard.press("Escape");
  await page.keyboard.press("Escape");
  report.checks.push(
    "file picker grants only the selected file through the real read-sources route",
  );
  const send = async (text, label = "发送") => {
    await page.getByRole("textbox", { name: "消息输入框" }).fill(text);
    await page.getByRole("button", { name: label, exact: true }).click();
    await page.waitForFunction(
      () => document.querySelector('[aria-label="消息输入框"]').value === "",
    );
  };
  await send("你好");
  await page.getByText("已收到：你好。这次聊天不改变后台工作。", { exact: true }).waitFor();
  await send("查询文件");
  await page.getByText("offline-fixture-marker", { exact: false }).first().waitFor();
  report.checks.push("real Chatbot tool routing reads authorized project file");
  await send("读取授权资料");
  await page.getByText("authorized-single-file-marker", { exact: false }).first().waitFor();
  await send("读取未授权相邻文件");
  await page.getByText("路径不在工作目录或用户授权资料目录内", { exact: false }).first().waitFor();
  assert.equal(await page.getByText("not-authorized-sibling-marker", { exact: false }).count(), 0);
  report.checks.push("single-file permission does not authorize a neighboring file");
  await send("预览文件");
  await page.getByRole("button", { name: "预览", exact: true }).click();
  await page
    .frameLocator('iframe[title="隔离文件预览"]')
    .getByRole("heading", { name: "集成预览" })
    .waitFor();
  const frame = page.frames().find((candidate) => candidate.url().includes("/preview/"));
  assert.equal(await frame.evaluate(() => Boolean(window.previewExecuted)), false);
  await page.keyboard.press("Escape");
  report.checks.push("real isolated HTML preview renders and blocks script");
  await send("外部打开文件");
  await page.getByRole("button", { name: "打开一次", exact: true }).click();
  await page.getByText("已打开", { exact: true }).waitFor();
  const opened = (await api(`/threads/${first}/opens`)).items.find(
    (item) => item.kind === "external",
  );
  assert.equal(opened.status, "opened");
  const repeat = await page.evaluate(
    async ({ thread, id, target }) => {
      const response = await fetch(`/api/threads/${thread}/opens`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, target, action: "approve" }),
      });
      return { status: response.status, data: await response.json() };
    },
    { thread: first, id: opened.id, target: opened.target },
  );
  assert.equal(repeat.status, 200);
  assert.equal(repeat.data.status, "opened");
  const nativeActions = (
    await fs.readFile(path.join(state, "fixture-native-actions.jsonl"), "utf8")
  )
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
  assert.equal(nativeActions.filter((action) => action.kind === "launch").length, 1);
  assert.equal((await api(`/threads/${first}/permission`)).effective, "ask");
  report.checks.push(
    "single-use approval persists once in the real backend without elevating work permissions",
  );
  const workStartsBefore = nativeActions.filter((action) => action.kind === "work_started").length;
  await send("帮我画鹈鹕骑自行车");
  await until(
    async () => (await api(`/conversations/${first}/snapshot`)).chat_status === "chat_complete",
  );
  const proposal = await api(`/conversations/${first}/snapshot`);
  assert.equal(proposal.work, null, "an unconfirmed initial request does not dispatch work");
  assert.ok(
    proposal.messages.some((message) => String(message.content).includes("按这个计划开始")),
  );
  const proposedActions = (
    await fs.readFile(path.join(state, "fixture-native-actions.jsonl"), "utf8")
  )
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
  assert.equal(
    proposedActions.filter((action) => action.kind === "work_started").length,
    workStartsBefore,
  );
  report.checks.push(
    "a new implementation request presents a plan and awaits confirmation without dispatching work",
  );
  await send("按这个计划开始");
  await page.getByRole("button", { name: "确认此方案并执行", exact: true }).waitFor();
  assert.equal((await api(`/conversations/${first}/snapshot`)).work, null);
  await page.getByRole("button", { name: "确认此方案并执行", exact: true }).click();
  report.checks.push("only the human confirmation control starts the saved delivery proposal");
  const started = await until(async () => {
    const value = await api(`/conversations/${first}/snapshot`);
    return value.work && ["running", "executing"].includes(value.work.status) && value;
  });
  assert.equal(started.repo, project, "selected workspace remains the authorized parent");
  assert.equal(started.work.repo, path.join(project, "pelican-bike"));
  assert.equal(await fs.stat(started.work.repo).then((value) => value.isDirectory()), true);
  const startActions = (await fs.readFile(path.join(state, "fixture-native-actions.jsonl"), "utf8"))
    .trim().split("\n").map((line) => JSON.parse(line));
  assert.equal(proposedActions.filter((action) => action.kind === "prepare").length, 0);
  assert.deepEqual(startActions.filter((action) => action.kind === "prepare").map((action) => action.repo),
    [started.work.repo]);
  await page
    .locator('details[aria-label="当前项目"]')
    .getByText(started.work.repo, { exact: true })
    .waitFor();
  const diagnosticReadsBefore = report.requests.filter((request) =>
    request.path.endsWith("/diagnostics"),
  ).length;
  await until(
    async () =>
      report.requests.filter((request) => request.path.endsWith("/diagnostics")).length >=
      diagnosticReadsBefore + 2,
  );
  const workActivity = page.locator('[data-testid="activity-group"][data-source="work"]').first();
  await workActivity.locator(":scope > summary").click();
  await workActivity.locator('[data-activity="thinking"] > summary').first().click();
  await workActivity.getByText("工作思考第一段", { exact: false }).waitFor();
  await workActivity.getByText("工作思考第一段，已确认要运行检查", { exact: true }).waitFor();
  await workActivity.getByText("printf 'offline-operation-marker'", { exact: true }).waitFor();
  assert.ok(
    !(await api(`/conversations/${first}/snapshot`)).messages.some((message) =>
      String(message.content).includes("工作思考第一段"),
    ),
  );
  report.checks.push(
    "current work reads saved activity continuously without opening a diagnostics panel",
  );
  await send("查询进度");
  await until(
    async () => (await api(`/conversations/${first}/snapshot`)).chat_status === "chat_complete",
  );
  const afterChat = await api(`/conversations/${first}/snapshot`);
  assert.equal(afterChat.work.task_id, started.work.task_id);
  assert.ok(["running", "executing"].includes(afterChat.work.status));
  report.checks.push("chat reads work status without pausing or replacing work");
  await page.getByRole("combobox", { name: "消息用途" }).selectOption("supplement");
  await send("补充：只处理必要文件", "发送补充要求");
  await until(async () => {
    const value = await api(`/conversations/${first}/snapshot`);
    return value.work?.reason?.includes("只处理必要文件");
  });
  report.checks.push("explicit supplement reaches original work mailbox");
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  await page.getByRole("button", { name: "继续任务", exact: true }).waitFor();
  const paused = await api(`/conversations/${first}/snapshot`);
  assert.equal(paused.work.status, "paused");
  const submitted = report.requests.filter(
    (request) => request.method === "POST" && request.path.endsWith("/commands"),
  ).length;
  await page.reload();
  await page.getByRole("button", { name: "继续任务", exact: true }).waitFor();
  assert.equal(
    report.requests.filter(
      (request) => request.method === "POST" && request.path.endsWith("/commands"),
    ).length,
    submitted,
  );
  report.checks.push("pause and refresh retain state without browser run or command replay");
  await until(async () => report.requests.some((request) => request.path.endsWith("/diagnostics")));
  await new Promise((resolve) => setTimeout(resolve, 500));
  const diagnosticReads = report.requests.filter((request) =>
    request.path.endsWith("/diagnostics"),
  ).length;
  await new Promise((resolve) => setTimeout(resolve, 1000));
  assert.equal(
    report.requests.filter((request) => request.path.endsWith("/diagnostics")).length,
    diagnosticReads,
  );
  assert.ok(
    report.requests
      .filter((request) => request.path.endsWith("/diagnostics"))
      .every((request) => request.method === "GET"),
  );
  assert.equal(await page.getByRole("button", { name: "刷新工作记录", exact: true }).count(), 0);
  report.checks.push(
    "paused work stops polling after its final read; display requests never execute commands",
  );
  await page.getByRole("button", { name: "继续任务", exact: true }).click();
  await page.getByRole("button", { name: "暂停", exact: true }).waitFor();
  await chooseProject(anotherProject);
  await page.waitForURL((url) => url.searchParams.get("threadId") !== first);
  const second = new URL(page.url()).searchParams.get("threadId");
  assert.equal((await api(`/conversations/${first}/snapshot`)).work.status, "paused");
  assert.equal((await api(`/conversations/${second}/snapshot`)).repo, anotherProject);
  await until(async () => (await page.getByTestId("activity-group").count()) === 0);
  assert.equal(await page.getByRole("button", { name: "查看工作记录", exact: true }).count(), 0);
  report.checks.push("workspace switch pauses old work and creates separate conversation");
  await page.getByRole("button", { name: "工具与设置", exact: true }).click();
  await page.getByRole("button", { name: "空间与环境", exact: true }).click();
  await page.getByRole("spinbutton", { name: "闲置缓存上限" }).waitFor();
  await until(
    async () => await page.getByRole("button", { name: "保存设置", exact: true }).isEnabled(),
  );
  assert.equal(await page.getByRole("spinbutton", { name: "闲置缓存上限" }).inputValue(), "20");
  assert.equal(await page.getByRole("spinbutton", { name: "数据盘预留" }).inputValue(), "5");
  await page.screenshot({ path: path.join(output, "resources.png"), fullPage: true });
  await page.keyboard.press("Escape");
  await page.keyboard.press("Escape");
  await page.screenshot({ path: path.join(output, "workspace.png"), fullPage: true });
  report.checks.push("real resource catalog displays 20 GiB and 5 GiB defaults");
  report.second_snapshot = await api(`/conversations/${second}/snapshot`);
  if (!(await page.getByRole("tab", { name: "回收站", exact: true }).isVisible()))
    await page.getByRole("button", { name: "对话记录", exact: true }).click();
  await page.getByRole("button", { name: "管理对话：新对话", exact: true }).click();
  await page.getByRole("menuitem", { name: "删除（可恢复）", exact: true }).click();
  await page.getByRole("tab", { name: "回收站", exact: true }).click();
  await page.getByRole("button", { name: "管理对话：新对话", exact: true }).click();
  await page.getByRole("menuitem", { name: "彻底删除", exact: true }).click();
  assert.equal(await page.getByRole("checkbox").isChecked(), false);
  await page.getByRole("button", { name: "彻底删除", exact: true }).click();
  await until(
    async () =>
      !(await api("/conversations")).conversations.some((value) => value.thread_id === second),
  );
  assert.equal((await fs.stat(anotherProject)).isDirectory(), true);
  report.checks.push(
    "trash and permanent deletion clean conversation records while preserving project directory",
  );
  assert.equal(report.errors.length, 0);
  assert.ok(
    report.requests.every((request) => !/\/(runs|control|activity)(\/|$)/.test(request.path)),
  );
  report.result = "passed";
  report.first_snapshot = await api(`/conversations/${first}/snapshot`);
  const stateFiles = await fs.readdir(state);
  assert.ok(stateFiles.includes("application.sqlite3"));
  assert.ok(!stateFiles.some((name) => name.startsWith("conversations.sqlite")));
  report.state_files = stateFiles;
  report.checks.push(
    "fresh product state uses application.sqlite3 without a mirrored conversations database",
  );
  process.stdout.write(
    "Real application browser integration passed; all models and work are offline fixtures.\n",
  );
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
  for (const child of processes) child.kill();
  await Promise.all(
    Object.entries(logs).map(([name, contents]) =>
      fs.writeFile(path.join(output, `${name}.log`), contents),
    ),
  );
  await fs.writeFile(path.join(output, "browser-result.json"), JSON.stringify(report, null, 2));
}
