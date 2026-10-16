"""使用 MonoPlanner 处理公开 SWE-bench 问题并导出预测。

官方测试和参考补丁只属于独立评分流程；
本模块只接收字段受限的公开任务清单。
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
import zipfile

from langchain_core.messages import messages_to_dict

from agent.settings import load_settings
from agent.execution.docker import DockerEnvironment
from eval.recording import RecordingModel, TokenLedger, redact, write_json
from agent.task.budget import StepBudget
from agent.model.client import ModelPolicy, create_model
from agent.execution.runtime_inventory import runtime_inventory, probe_runtime
from agent.source.snapshots import _selected_files, _safe_file, _inspect_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FIELDS = {"instance_id", "repo", "base_commit", "problem_statement"}


# ============================================================
# 评测源码和依赖版本归档
# ============================================================
def archive_source(directory):
    """保留本次评测使用的准确实现，不包含 .env 或第三方数据。"""
    files = list(PROJECT_ROOT.glob("*.py"))
    files.extend(PROJECT_ROOT.glob("requirements*.txt"))
    files.extend(path for path in (PROJECT_ROOT / 'scripts').glob('*.ps1'))
    files.extend((PROJECT_ROOT / 'launcher').glob('*.cs'))
    for folder, directories, names in os.walk(PROJECT_ROOT / 'web', followlinks=False):
        directories[:] = [
            name
            for name in directories
            if name not in {'node_modules', '.next', '.git'}
            and not Path(folder, name).is_symlink()
            and not Path(folder, name).is_junction()
        ]
        files.extend(
            path
            for name in names
            if (path := Path(folder, name)).is_file()
            and path.suffix
            in {'.ts', '.tsx', '.js', '.json', '.css', '.md', '.yaml', '.yml', '.svg'}
            and not path.name.endswith('.tsbuildinfo')
        )
    files.extend(
        PROJECT_ROOT / name
        for name in ('README.md', 'web/LICENSE', 'web/.env.example')
    )
    for folder in ("agent", "tests", "resources"):
        files.extend(
            path
            for path in (PROJECT_ROOT / folder).rglob("*")
            if path.is_file() and path.suffix in {".py", ".txt", ".json"}
        )
    files.extend((PROJECT_ROOT / "eval").glob("*.py"))
    for folder in ("analysis",):
        files.extend(
            path
            for path in (PROJECT_ROOT / "eval" / folder).rglob("*")
            if path.is_file() and path.suffix in {".py", ".json"}
        )
    hashes = {}
    with zipfile.ZipFile(directory / "evaluated-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, relative)
    write_json(directory / "source-sha256.json", hashes)
    write_json(
        directory / "package-versions.json",
        {
            item.metadata["Name"]: item.version
            for item in metadata.distributions()
            if item.metadata["Name"]
        },
    )


# ============================================================
# 公开任务清单的校验
# ============================================================
def load_tasks(path):
    tasks = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Public task manifest must be a non-empty list.")
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != PUBLIC_FIELDS:
            raise ValueError("Only instance_id/repo/base_commit/problem_statement are allowed.")
        if any(not isinstance(value, str) or not value.strip() for value in task.values()):
            raise ValueError("Public task fields must be non-empty strings.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+", task["instance_id"]):
            raise ValueError("Unsafe instance_id.")
        if not re.fullmatch(r"[0-9a-f]{40}", task["base_commit"]):
            raise ValueError("base_commit must be a full commit hash.")
        if task["instance_id"] in seen:
            raise ValueError("Duplicate instance_id.")
        seen.add(task["instance_id"])
    return tasks


def run_logged(arguments, log_prefix, cwd=None, timeout=180, env=None):
    log_prefix = Path(log_prefix)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(log_prefix.with_suffix(".command.json"), {"argv": arguments, "cwd": str(cwd)})
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    with (
        log_prefix.with_suffix(".stdout.log").open("wb") as out,
        log_prefix.with_suffix(".stderr.log").open("wb") as err,
    ):
        process = subprocess.run(
            arguments, cwd=cwd, stdout=out, stderr=err, timeout=timeout, env=env, **options
        )
    if process.returncode:
        raise RuntimeError(f"Command exited {process.returncode}; see {log_prefix}.stderr.log")
    return log_prefix.with_suffix(".stdout.log").read_bytes()


def git_args(repo, *arguments):
    return [
        "git",
        "-c",
        f"safe.directory={Path(repo).resolve().as_posix()}",
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.filemode=false",
        "-c",
        "core.longpaths=true",
        *arguments,
    ]


# ============================================================
# 提取并记录原始任务源码
# ============================================================
def extract_checkout(image, repo, logs, tag):
    repo.mkdir(parents=True, exist_ok=False)
    container = (
        run_logged(
            [
                "docker",
                "create",
                "--name",
                f"swe-source-{tag}",
                "--network",
                "none",
                "--entrypoint",
                "/bin/true",
                image,
            ],
            logs / "source-create",
        )
        .decode()
        .strip()
    )
    # 同时保留已停止的提取容器和本机源码副本。
    run_logged(
        ["docker", "cp", f"{container}:/testbed/.", str(repo)], logs / "source-copy", timeout=300
    )
    return container


def prepare_base_checkout(repo, base_commit, logs):
    """仅规范化副本中的安装阶段 Git 元数据，不替换任何源码。

    官方镜像可能有专门记录 chmod +x 的额外提交；即使 filemode=false，
    与该索引比较仍可能出现大量只有权限位不同的文件。
    mixed reset 只调整 HEAD 和索引；实际源码差异仍会被后续检查拒绝。
    原始提取容器始终保留。
    """
    run_logged(git_args(repo, "rev-parse", "HEAD"), logs / "image-head", repo)
    run_logged(git_args(repo, "ls-files", "--stage", "-z"), logs / "image-index", repo)
    run_logged(git_args(repo, "config", "core.filemode", "false"), logs / "filemode-config", repo)
    run_logged(
        git_args(repo, "config", "core.autocrlf", "false"), logs / "line-ending-config", repo
    )
    run_logged(git_args(repo, "reset", "--mixed", base_commit), logs / "base-index", repo)
    baseline = run_logged(git_args(repo, "diff", base_commit, "--"), logs / "baseline-diff", repo)
    if baseline.strip():
        raise RuntimeError("Image source content differs from the specified task base commit.")


def record_initial_workspace(repo, task, image_id, logs):
    """在模型采取任何行动前，记录独立提取的源码起点。"""
    inventory = runtime_inventory(repo)
    source = {}
    for relative, skip, tracked in _selected_files(repo)[1]:
        if skip or not (repo / relative).is_file():
            continue
        digest, private = _inspect_file(_safe_file(repo, relative))
        if not private:
            source[relative.as_posix()] = digest
    record = {
        "instance_id": task["instance_id"],
        "base_commit": task["base_commit"],
        "image_id": image_id,
        "source_files": source,
        "runtime": inventory,
        "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        "public_task_sha256": hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
        "fresh_extraction": True,
        "initial_history": "empty",
        "network": "none",
    }
    write_json(logs / "initial-workspace.json", record)
    return inventory


# ============================================================
# 导出最终文件树的补丁
# ============================================================
def export_patch(repo, base_commit, directory, baseline_untracked=(), environment=None):
    """根据最终文件树生成补丁；不受信任的任务 Git 命令始终在容器内执行。

    仅离线测试中受信任的临时仓库允许使用本机后备路径。
    独立索引不会修改任务原来的暂存区。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / "patch.index"
    if index.exists():
        raise FileExistsError("Refusing to overwrite an earlier patch index.")

    def git(arguments, name, temporary_index=True):
        log_prefix = directory / name
        if environment is None:
            env = dict(os.environ)
            env.pop("GIT_INDEX_FILE", None)
            if temporary_index:
                env["GIT_INDEX_FILE"] = str(index.resolve())
            return run_logged(
                git_args(repo, "--literal-pathspecs", *arguments), log_prefix, repo, env=env
            )
        # 不能在本机运行受 Agent 控制的 Git 配置或属性。
        # 新容器不接收本机环境变量或凭据。
        index_option = (
            [f"GIT_INDEX_FILE={environment.host_to_container(index)}"]
            if temporary_index
            else ["-u", "GIT_INDEX_FILE"]
        )
        command = [
            "/usr/bin/env",
            *index_option,
            "git",
            "--literal-pathspecs",
            "-c",
            "safe.directory=/testbed",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.filemode=false",
            *arguments,
        ]
        write_json(
            log_prefix.with_suffix(".command.json"),
            {
                "argv": command,
                "cwd": "/testbed",
                "backend": "docker",
            },
        )
        stdout_path, stderr_path = (
            log_prefix.with_suffix(".stdout.log"),
            log_prefix.with_suffix(".stderr.log"),
        )
        outcome = environment.run_process(
            command, stdout_path=stdout_path, stderr_path=stderr_path, timeout=180, shell=False
        )
        if outcome.get("error") or outcome.get("timed_out") or outcome.get("exit_code") != 0:
            raise RuntimeError(f"Container Git failed; see {stderr_path}")
        return stdout_path.read_bytes()

    def filenames(data):
        return {name for name in data.decode("utf-8").split("\0") if name}

    def excluded_new_file(name):
        parts = Path(name).parts
        return (
            name in baseline_untracked
            or any(part in {"__pycache__", ".pytest_cache"} for part in parts)
            or Path(name).name == ".env"
            or Path(name).suffix in {".pyc", ".pem", ".key"}
        )

    base_files = filenames(
        git(
            ["ls-tree", "-r", "--name-only", "-z", base_commit], "base-files", temporary_index=False
        )
    )
    original_files = filenames(
        git(["ls-files", "--cached", "-z"], "original-index-files", temporary_index=False)
    )
    deleted_files = filenames(
        git(["ls-files", "--deleted", "-z"], "deleted-worktree-files", temporary_index=False)
    )
    git(["read-tree", base_commit], "index-base")
    git(["add", "--all", "--", "."], "index-add")
    excluded = set()
    # 新索引会遗漏由 git add -f 强制加入的忽略文件，需要保留这些记录。
    for number, name in enumerate(sorted(original_files - base_files - deleted_files), 1):
        if excluded_new_file(name):
            excluded.add(name)
            continue
        git(["add", "--force", "--", name], f"include-staged-{number:03d}")
    candidates = filenames(git(["ls-files", "-z"], "index-files"))
    for name in sorted(candidates):
        if name in base_files:
            continue
        if excluded_new_file(name):
            excluded.add(name)
            git(["rm", "--cached", "--ignore-unmatch", "--", name], f"exclude-{len(excluded):03d}")
    write_json(directory / "excluded-new-files.json", sorted(excluded))
    patch = git(["diff", "--cached", "--binary", base_commit, "--"], "patch-export")
    (directory / "model.patch").write_bytes(patch)
    write_json(
        directory / "patch.json",
        {
            "bytes": len(patch),
            "sha256": hashlib.sha256(patch).hexdigest(),
            "base_commit": base_commit,
        },
    )
    return patch.decode("utf-8")


def save_predictions(directory, results, model_name):
    records = [
        {
            "instance_id": item["instance_id"],
            "model_name_or_path": model_name,
            "model_patch": item.get("model_patch", ""),
        }
        for item in results
    ]
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    write_json(
        directory / "summary.json",
        [{key: value for key, value in item.items() if key != "model_patch"} for item in results],
    )


def create_task_controller(recorder, environment, directory, args):
    """每道评测题都使用同一个任务执行器。"""
    from agent.task.runner import TaskRunner

    return TaskRunner(
        recorder,
        environment,
        directory / "plan-runtime",
        max_rounds=args.max_rounds,
        task_token_limit=args.task_token_limit,
        task_timeout=args.task_timeout,
    )


# ============================================================
# 单道题的完整执行流程
# ============================================================
def run_instance(task, image, directory, model, ledger, args, secrets):
    started = time.monotonic()
    result = {
        "instance_id": task["instance_id"],
        "agent_status": "not_started",
        "official_resolved": None,
        "model_patch": "",
        "errors": [],
        "workflow": "task",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "task.json", task)
    before = asdict(ledger)
    budget = StepBudget(
        max_steps=args.max_steps,
        max_tokens=args.task_token_limit,
        timeout_seconds=args.task_timeout,
    )
    repo, logs = directory / "repo", directory / "setup"
    environment, baseline_untracked, ready = None, [], False
    tag = f"{task['instance_id'].replace('__', '-')}-{uuid.uuid4().hex[:8]}"
    try:
        image_data = run_logged(["docker", "image", "inspect", image], logs / "image-inspect")
        result["image_id"] = json.loads(image_data)[0]["Id"]
        result["requested_image"] = image
        image = result["image_id"]  # 子快照的镜像不能随可变标签发生变化。
        result["source_container"] = extract_checkout(image, repo, logs, tag)
        run_logged(
            git_args(repo, "cat-file", "-e", f"{task['base_commit']}^{{commit}}"),
            logs / "base-exists",
            repo,
        )
        prepare_base_checkout(repo, task["base_commit"], logs)
        inventory = record_initial_workspace(repo, task, image, logs)
        baseline_untracked = (
            run_logged(
                git_args(repo, "ls-files", "--others", "--exclude-standard", "-z"),
                logs / "baseline-untracked",
                repo,
            )
            .decode()
            .split("\0")
        )
        ready = True
        environment = DockerEnvironment(
            repo,
            image,
            f"swe-agent-{tag}",
            timeout=args.command_timeout,
            artifact_root=directory / "runtime",
        )
        environment.runtime_files = tuple(inventory["files"])
        environment.runtime_packages = tuple(inventory["packages"])
        environment.start()
        if inventory["packages"]:
            runtime = probe_runtime(
                environment, inventory["packages"], environment.artifact_root / "preflight"
            )
            if not runtime["ok"]:
                raise RuntimeError(
                    "Fresh task workspace failed import preflight; see runtime/preflight/report.json"
                )
            environment.runtime_capabilities = runtime["checkers"]
        run_logged(["docker", "inspect", environment._container_id], logs / "container-inspect")
        # 不能挂载题库、参考补丁、官方测试、模型密钥或 Docker Socket。
        recorder = RecordingModel(
            model,
            directory / "model-calls",
            ledger,
            Path(args.run_dir) / "token-ledger.json",
            args.max_output_tokens,
            secrets,
            deadline=time.monotonic() + args.task_timeout,
            task_token_limit=args.task_token_limit,
            report_token_reserve=24_000,
            report_time_reserve=30,
        )
        controller = create_task_controller(recorder, environment, directory, args)
        outcome = controller.run(
            [],
            "Please implement a code fix for this repository issue:\n\n"
            + task["problem_statement"],
            budget,
        )
        serializable = {
            **outcome,
            "messages": messages_to_dict(outcome["messages"]),
            "budget": asdict(budget),
        }
        write_json(directory / "agent-result.json", redact(serializable, secrets))
        result.update(
            agent_status=outcome["status"],
            stop_reason=outcome.get("stop_reason"),
            internal_verification=outcome.get("verification", {}).get("status"),
            rounds=len(outcome.get("rounds", [])),
            report_status=outcome.get("report_status"),
            report_used_steps=budget.report_used_steps,
        )
    except Exception as exc:
        result["agent_status"] = "evaluation_error"
        result["errors"].append(redact(f"{type(exc).__name__}: {exc}", secrets))
    finally:
        producer_stopped = environment is None
        if environment is not None:
            try:
                environment.close()
                producer_stopped = True
            except Exception as exc:
                result["errors"].append(f"Container stop failed: {exc}")
        if ready and producer_stopped:
            try:
                # Agent 容器停止后，再启动新的镜像进程。
                # Git 过滤器和配置由任务控制，绝不能在本机运行。
                with DockerEnvironment(
                    repo,
                    image,
                    f"swe-export-{tag}",
                    timeout=180,
                    artifact_root=directory / "export-runtime",
                ) as exporter:
                    result["model_patch"] = export_patch(
                        repo,
                        task["base_commit"],
                        exporter.artifact_root / "patch",
                        baseline_untracked,
                        environment=exporter,
                    )
            except Exception as exc:
                result["errors"].append(f"Patch export failed: {exc}")
        elif ready:
            result["errors"].append("Patch export skipped: task container stop was not confirmed.")
        result.update(
            elapsed_seconds=round(time.monotonic() - started, 3),
            logical_model_calls=budget.used_steps,
            sent_model_calls=ledger.calls - before["calls"],
            locally_blocked_calls=ledger.locally_blocked_calls - before["locally_blocked_calls"],
            local_guard_reason=(
                ledger.last_guard_reason
                if ledger.locally_blocked_calls > before["locally_blocked_calls"]
                else None
            ),
            reported_tokens=ledger.reported_tokens - before["reported_tokens"],
            unknown_usage_calls=ledger.unknown_calls - before["unknown_calls"],
            patch_bytes=len(result["model_patch"].encode("utf-8")),
        )
        write_json(directory / "result.json", redact(result, secrets))
    return result


# ============================================================
# 命令行参数与批量运行入口
# ============================================================
def _argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument(
        "--token-limit",
        type=int,
        default=3_000_000,
        help="Evaluation token ceiling; 0 disables this ceiling while retaining accounting",
    )
    parser.add_argument(
        "--task-token-limit",
        type=int,
        default=700_000,
        help="Per-task token ceiling; 0 disables this ceiling while retaining accounting",
    )
    parser.add_argument("--max-output-tokens", type=int, default=ModelPolicy().max_output_tokens)
    parser.add_argument("--command-timeout", type=int, default=120)
    parser.add_argument("--task-timeout", type=int, default=1800)
    return parser


def parse_args(argv=None):
    parser = _argument_parser()
    args = parser.parse_args(argv)
    for key in ("token_limit", "task_token_limit"):
        if getattr(args, key) < 0:
            parser.error(f"{key} must be nonnegative (0 means unlimited)")
        if getattr(args, key) == 0:
            setattr(args, key, None)
    for key in ("max_steps", "max_rounds", "max_output_tokens", "command_timeout", "task_timeout"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    tasks = load_tasks(args.tasks)
    if args.instance_id:
        if set(args.instance_id) - {task["instance_id"] for task in tasks}:
            _argument_parser().error("Unknown instance id")
        tasks = [task for task in tasks if task["instance_id"] in args.instance_id]
    image_rows = json.loads(Path(args.images).read_text(encoding="utf-8-sig"))
    images = {item["instance_id"]: item["image"] for item in image_rows}
    if any(task["instance_id"] not in images for task in tasks):
        _argument_parser().error("Missing image mapping")
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    args.run_dir = str(directory)
    archive_source(directory)
    settings = load_settings()
    policy = ModelPolicy(max_output_tokens=args.max_output_tokens)
    model = create_model(settings, policy)
    ledger = TokenLedger(limit=args.token_limit)
    write_json(
        directory / "config.json",
        {
            **vars(args),
            "model": settings["model_name"],
            "temperature": 0,
            "sdk_retries": 0,
            "api_timeout": 90,
            "model_policy": asdict(policy),
            "report_step_reserve": 1,
            "report_token_reserve": 24_000,
            "report_time_reserve": 30,
            "tasks": tasks,
            "images": images,
            "official_tests_visible_to_agent": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    results = []
    model_label = f"coding-agent-{settings['model_name']}-{directory.name}"
    for task in tasks:
        print(f"Starting {task['instance_id']}", flush=True)
        result = run_instance(
            task,
            images[task["instance_id"]],
            directory / task["instance_id"],
            model,
            ledger,
            args,
            (settings["api_key"],),
        )
        results.append(result)
        save_predictions(directory, results, model_label)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "instance_id",
                        "agent_status",
                        "logical_model_calls",
                        "reported_tokens",
                        "patch_bytes",
                    )
                }
            ),
            flush=True,
        )
    write_json(directory / "token-ledger.json", asdict(ledger))
    # Agent 的停止状态作为结果数据保存；只有调度故障才让此命令失败。
    return int(any(item["errors"] for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
