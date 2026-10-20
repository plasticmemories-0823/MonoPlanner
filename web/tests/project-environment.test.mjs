import assert from "node:assert/strict";
import test from "node:test";
import { ProjectEnvironmentMonitor } from "../src/lib/project-environment.ts";

const status = (state, fields = {}) => ({
  mode: "auto",
  status: state,
  stage: "",
  message: state,
  ...fields,
});
const response = (value, code = 200) => Response.json(value, { status: code });
const pause = () => new Promise((resolve) => setTimeout(resolve, 30));
const deferred = () => {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
};
function fixture(t, fetcher, timeout = 10000) {
  const states = [],
    errors = [],
    busy = [];
  const monitor = new ProjectEnvironmentMonitor(
    "/api",
    "C:\\user project",
    {
      status: (value) => states.push(value),
      error: (value) => errors.push(value),
      busy: (value) => busy.push(value),
    },
    fetcher,
    5,
    timeout,
  );
  t.after(() => monitor.dispose());
  return { monitor, states, errors, busy };
}

test("browser fetch is called without a custom receiver", async (t) => {
  const { monitor, states, errors } = fixture(t, async function () {
    assert.equal(this, undefined, "native browser fetch rejects a monitor receiver");
    return response(status("idle"));
  });
  await monitor.refresh();
  assert.equal(states[0].status, "idle");
  assert.equal(errors.at(-1), "");
});

test("opening an idle project only reads status and never installs or enumerates images", async (t) => {
  const calls = [];
  const { monitor, states } = fixture(t, async (url, options) => {
    calls.push({ url, options });
    return response(status("idle"));
  });
  await monitor.refresh();
  await pause();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].options.method, undefined);
  assert.equal(
    new URL(calls[0].url, "http://localhost").searchParams.get("repo"),
    "C:\\user project",
  );
  assert.equal(states[0].status, "idle");
});

test("failed project retry stays bound to the actual project, not its authorized parent", async (t) => {
  const calls = [];
  const actualProject = "E:\\workspace\\pelican";
  const monitor = new ProjectEnvironmentMonitor(
    "/api",
    actualProject,
    { status() {}, error() {}, busy() {} },
    async (url, options) => {
      calls.push({ url, options });
      return response(status(options.method === "POST" ? "ready" : "failed"));
    },
  );
  t.after(() => monitor.dispose());
  await monitor.refresh(true);
  assert.equal(calls.length, 1);
  assert.equal(new URL(calls[0].url, "http://localhost").searchParams.get("repo"), actualProject);
  await monitor.change("prepare");
  assert.deepEqual(JSON.parse(calls[1].options.body), { repo: actualProject, action: "prepare" });
  assert.equal(calls.filter((call) => call.options.method === "POST").length, 1);
});

test("preparation polling stays serial and stops when ready", async (t) => {
  const second = deferred();
  let calls = 0;
  const { monitor, states } = fixture(t, async () => {
    return ++calls === 1 ? response(status("preparing")) : second.promise;
  });
  await monitor.refresh();
  await pause();
  assert.equal(calls, 2);
  await pause();
  assert.equal(calls, 2, "an unfinished read must not create another poll");
  second.resolve(response(status("ready")));
  await pause();
  assert.equal(calls, 2);
  assert.deepEqual(
    states.map((s) => s.status),
    ["preparing", "ready"],
  );
});

test("prepare supersedes stale reads and duplicate clicks join the same request", async (t) => {
  const first = deferred(),
    post = deferred();
  const calls = [];
  const { monitor, states, busy } = fixture(t, async (url, options) => {
    calls.push({ url, options });
    return options.method === "POST" ? post.promise : first.promise;
  });
  const read = monitor.refresh();
  const prepare = monitor.change("prepare");
  await monitor.change("prepare");
  assert.equal(calls.length, 2);
  assert.equal(calls[0].options.signal.aborted, true);
  assert.deepEqual(JSON.parse(calls[1].options.body), {
    repo: "C:\\user project",
    action: "prepare",
  });
  post.resolve(response(status("ready")));
  await prepare;
  first.resolve(response(status("idle")));
  await read;
  assert.deepEqual(
    states.map((s) => s.status),
    ["ready"],
  );
  assert.deepEqual(busy, [true, false]);
});

test("switching folders or closing dialog aborts requests and cannot apply late updates", async (t) => {
  const pending = deferred();
  let signal;
  const { monitor, states } = fixture(t, async (_url, options) => {
    signal = options.signal;
    return pending.promise;
  });
  const read = monitor.refresh();
  monitor.dispose();
  assert.equal(signal.aborted, true);
  pending.resolve(response(status("preparing")));
  await read;
  await pause();
  assert.deepEqual(states, []);
});

test("failed saves preserve the current environment and do not retry the mutation", async (t) => {
  let calls = 0;
  const { monitor, states, errors, busy } = fixture(t, async () => {
    return ++calls === 1 ? response(status("ready")) : response({ error: "保存失败" }, 409);
  });
  await monitor.refresh();
  await monitor.saveManual("sha256:example", {
    python: "/usr/bin/python3",
    conda_env: "",
  });
  await pause();
  assert.equal(calls, 2);
  assert.equal(states.length, 1);
  assert.equal(states[0].mode, "auto");
  assert.equal(errors.at(-1), "保存失败");
  assert.deepEqual(busy, [true, false]);
});

test("manual save reads back the selected binding without starting preparation", async (t) => {
  const calls = [];
  const binding = {
    image: "sha256:legacy",
    runtime: {
      python: "/opt/miniconda3/envs/testbed/bin/python",
      conda_env: "testbed",
    },
  };
  const { monitor, states } = fixture(t, async (url, options) => {
    calls.push({ url, options });
    return response(
      options.method === "POST"
        ? binding
        : status("ready", { mode: "manual", environment: binding }),
    );
  });
  await monitor.saveManual(binding.image, binding.runtime);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].url, "/api/workspace/execution");
  assert.equal(calls[1].options.method, undefined);
  assert.deepEqual(states[0].environment, binding);
  assert.equal(states[0].mode, "manual");
});

test("reconnect only retries a read, and preparation failures require explicit retry", async (t) => {
  const calls = [];
  const { monitor, states, errors } = fixture(t, async (_url, options) => {
    calls.push(options);
    if (calls.length === 1) throw new Error("offline");
    return response(status("failed", { error: "依赖下载失败" }));
  });
  await monitor.refresh();
  assert.equal(errors.at(-1), "offline");
  await pause();
  assert.equal(calls.length, 1);
  await monitor.refresh();
  await pause();
  assert.equal(calls.length, 2);
  assert.ok(calls.every((call) => !call.method));
  assert.equal(states[0].status, "failed");
});

test("explicit verification checks resources once while polling remains a plain read", async (t) => {
  const calls = [];
  const { monitor, states } = fixture(t, async (url, options) => {
    calls.push({ url: new URL(url, "http://localhost"), options });
    return response(status(calls.length === 1 ? "preparing" : "failed"));
  });
  await monitor.refresh(true);
  await pause();
  assert.equal(calls.length, 2);
  assert.equal(calls[0].url.searchParams.get("verify"), "1");
  assert.equal(calls[1].url.searchParams.has("verify"), false);
  assert.ok(calls.every(({ options }) => !options.method));
  assert.equal(states.at(-1).status, "failed");
});

test("timed out preparation releases busy state and never resends the operation", async (t) => {
  const calls = [];
  const { monitor, errors, busy } = fixture(
    t,
    async (_url, options) => {
      calls.push(options);
      return new Promise((resolve, reject) => {
        options.signal.addEventListener("abort", () => reject(new Error("aborted")), {
          once: true,
        });
      });
    },
    15,
  );
  await monitor.change("prepare");
  await pause();
  assert.equal(calls.length, 1);
  assert.deepEqual(busy, [true, false]);
  assert.match(errors.at(-1), /结果尚未确认/);
  assert.equal(calls[0].signal.aborted, true);
});
