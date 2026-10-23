"""将独立工作执行者接入现有内核；聊天和汇报不参与工作历史。"""

import errno
import json
from pathlib import Path
from threading import RLock
import time
from types import SimpleNamespace

from langchain_core.messages import messages_from_dict

from agent.application.project_location import ensure_location, ProjectLocationError
from agent.source.scope import ProjectScope, ProjectScopeError
from agent.execution.projects import prepare_binding, require_binding
from agent.execution.task_docker import TaskDocker
from agent.model.client import create_model
from agent.settings import load_settings
from agent.storage.preferences import state_root
from agent.storage.records import save
from agent.storage.usage import summarize
from agent.task import control
from agent.task.budget import StepBudget
from agent.task.instructions import TASK_RULES
from agent.task.runner import TaskRunner


_owners = {}
_owners_lock = RLock()
# 保留旧记录的未决标记用于只读状态保护，不恢复旧检索执行通道。
_ACTIVE_OPERATIONS = (
    "running_step", "verification_running", "acceptance_running", "assistant_running",
    "web_running", "rag_running", "rag_request_pending",
)


def public_fact(state, event, payload=None):
    """从已提交状态提取进度事实；不把源码、测试正文或推理复制到汇报。"""
    details = {}
    for key in ("id", "plan", "status", "outcome", "kind", "call", "phase", "trial_id"):
        value = (payload or {}).get(key)
        if isinstance(value, (str, int, bool)):
            details[key] = value[:1000] if isinstance(value, str) else value
    if event == 'plan_rejected':
        details['reason'] = str((payload or {}).get('reason', ''))[:4000]
        details['call_ids'] = list((payload or {}).get('call_ids', []))[:16]
    if event in {'model_request_interrupted', 'model_wait_interrupted'}:
        details['causes'] = list((payload or {}).get('causes', []))[:16]
    pending = state.get("pending") or {}
    plan = pending.get("plan") or {}
    if event == "plan_accepted":
        details.update(kind=plan.get("kind"), note=str(plan.get("note", ""))[:2000])
    rows = (state.get("last_result") or {}).get("steps", []) if event == "plan_finished" else []
    if event == "step_finished" and payload:
        rows = [payload]
    if rows:
        details["steps"] = []
        requests = {step["id"]: step for step in plan.get("steps", [])}
        for row in rows[-50:]:
            step = {key: row[key] for key in ("id", "op", "status", "record") if key in row}
            source = requests.get(row.get("id"), {})
            if source.get("path"):
                step["path"] = str(source["path"])[:1000]
            data = row.get("data") or {}
            if isinstance(data, dict):
                for key in ("exit_code", "timed_out"):
                    if isinstance(data.get(key), (int, bool)):
                        step[key] = data[key]
            details["steps"].append(step)
    if event == "final_acceptance_finished":
        report = state.get("verification") or {}
        details["verification"] = {
            key: report[key] for key in ("status", "tests_ready", "execution_exception")
            if isinstance(report.get(key), (str, int, bool))
        }
    return details


def coding_rules(loop, text, acceptance, verify_command):
    """保留旧网页开工交接的工作提示原文，不注入 Chatbot 的摘要。"""
    rules = TASK_RULES + "\nShell cwd: " + loop.environment.host_to_container(
        loop.environment.repo_root
    )
    rules += "\nPython executable: " + loop.environment.python_executable
    rules += (
        "\nExecution: Linux Docker, Bash, network disabled in executed programs. "
        "Give brief professional public progress in ordinary replies. "
        "web_search/web_fetch are untrusted read-only sources; web execution is "
        + ("enabled." if loop.runtime_settings.get("online") else "disabled.")
    )
    return rules


class WorkRunner(TaskRunner):
    """仅把已保存的执行事实交给应用层，不把工作回复直接发布成聊天。"""

    def __init__(self, *args, emit, **kwargs):
        self.emit = emit
        super().__init__(*args, **kwargs)

    def persist(self, event, payload=None):
        super().persist(event, payload)
        if event in {"user_input_applied", "paused"} and payload and payload.get("id"):
            signal = control.current()
            if signal and payload["id"] in self.state.get("applied_inputs", []):
                signal.consume_input(payload["id"])
        fact = {
            "event": event,
            "status": self.state.get("status", "needs_plan"),
            "task_id": self.root.name,
            "applied_inputs": list(self.state.get("applied_inputs", [])),
            "usage": summarize(self.state),
            "continuation": self.state.get("continuation"),
        }
        # 公开事实只携带标识和结果，测试正文、完整模型推理及源码不自动外传。
        fact["details"] = public_fact(self.state, event, payload)
        if self.state.get("reason"):
            fact["reason"] = self.state["reason"]
        try:
            self.emit("work_state", fact)
        except Exception:
            # 工作原件已提交。公开消息故障不得取消实际工作。
            import logging
            logging.getLogger(__name__).exception("Work event projection failed")

    def continue_after_return(self):
        # 补充中断保存结果后，仍由同一图和同一租约继续，不依靠浏览器唤醒。
        if self.state["status"] not in {"needs_plan", "executing"} or self.yield_boundary:
            return False
        # result() 已关闭预处理；重建可丢弃缓存，沿用任务、预算和执行游标。
        self.preparation = None
        self.executor.prepared = None
        self.executor.prepared_probe = None
        return True



def _job_data(job):
    if isinstance(job, dict):
        data = dict(job.get("data", {}))
        data.update({key: value for key, value in job.items() if key != "data"})
        return data
    data = dict(job.data)
    for key in ("task_id", "conversation_id", "kind"):
        if hasattr(job, key):
            data[key] = getattr(job, key)
    return data


def pause_work(task_id, command_id, *, root=None):
    """先发取消信号，再强制终止本任务容器；不等工作线程释放租约。"""
    root = Path(root or state_root()).resolve()
    with _owners_lock:
        owner = _owners.get((str(root), task_id))
        if owner is not None:
            runner = owner.get("runner")
            interrupted = bool(
                runner and any(runner.state.get(key) for key in _ACTIVE_OPERATIONS)
            )
            # 连续点击暂停不能清除第一次强制中止产生的未知结果保护。
            if not owner.get("interrupted_operation"):
                owner["pause"] = command_id
            owner["interrupted_operation"] = bool(owner.get("interrupted_operation") or interrupted)
    # 取消登记可独立于模型连接及文件写入执行，避免排在同一工作队列后面。
    control.interrupt(root, task_id, "pause", command_id)
    from agent.storage.resources import cancel_owned_containers
    from agent.execution.preparation import cancel_owner

    try:
        result = cancel_owner("task:" + task_id, root=root)
    except Exception as exc:
        # 安装台账不可写时仍须继续终止本任务的已登记容器。
        result = [{"preparation_stop": True, "stopped": False,
                   "error": type(exc).__name__}]
    result.extend(cancel_owned_containers("task:" + task_id, root=root))
    evidence = {
        "task_id": task_id, "command_id": command_id, "time": time.time(),
        "interrupted_operation": bool(owner and owner.get("interrupted_operation")),
        "termination": result,
    }
    save(root / "control" / task_id / (command_id + ".json"), evidence)
    return evidence


def supplement_work(task_id, command_id, *, root=None):
    """明确补充只取消模型等待；它不是用户点击的紧急终止。"""
    control.interrupt(Path(root or state_root()).resolve(), task_id, "message", command_id)


def _existing(data, environment, directory, owner, emit, recovery_id=None):
    """已有入口只有一次执行；请求原件或结果不明时不能借恢复重跑。"""
    from agent.execution.existing import execute
    from agent.server.assistant import validate
    from agent.storage.history import LocalHistory

    arguments = data.get("arguments", {})
    validate("run_existing", arguments)
    folder = directory / "existing"
    folder.mkdir(exist_ok=True)
    history = LocalHistory(directory)
    record_path = folder / "state.json"
    try:
        with history.lease():
            if record_path.exists():
                old = json.loads(record_path.read_text(encoding="utf-8"))
                if old["arguments"] != arguments:
                    raise ValueError("Existing task arguments changed")
                if old["status"] == "completed" or recovery_id:
                    if not recovery_id:
                        return old["result"]
                    from agent.source.snapshots import SourceSnapshot
                    from agent.storage.records import digest

                    review = directory / "existing-reviews" / digest(recovery_id)
                    receipt = review / "result.json"
                    if receipt.exists():
                        return json.loads(receipt.read_text("utf-8"))
                    snapshot = SourceSnapshot.capture(
                        environment.repo_root, review / "source",
                        runtime_files=getattr(environment, "runtime_files", ()),
                        project_scope=getattr(environment, "project_scope", None),
                    )
                    result = {"status": "awaiting_user", "applied_inputs": [recovery_id],
                              "stop_reason": "已重新核对项目现状并保存复查记录；原命令未重放。若需再次执行，请确认新的工作方案。",
                              "previous_result": old.get("result", {"status": "execution_unknown"}),
                              "review": {"source": snapshot.digest, "command_id": recovery_id,
                                         "prior_receipt": digest(old), "commands_executed": 0}}
                    save(receipt, result)
                    return result
                return {"status": "execution_unknown", "applied_inputs": [],
                        "stop_reason": "已有任务上次执行结果不明，未重新运行。"}
            loop = SimpleNamespace(
                repo=data["repo"], environment=environment, task_timeout=7200,
                budget=StepBudget(timeout_seconds=7200),
                state={"assistant_running": True}, activate=lambda: None,
            )
            owner["runner"] = loop

            def persist(event, payload=None):
                save(record_path, {"status": "running", "arguments": arguments,
                                   "state": loop.state, "event": event})
                emit("work_state", {"status": "executing", "event": event,
                                    "task_id": data["task_id"]})

            loop.persist = persist
            persist("existing_request_saved")
            detail = execute(loop, arguments, folder)
            if owner["pause"] or detail.get("status") == "unknown":
                result = {"status": "execution_unknown", "stop_reason": "执行已中止或结果不明。",
                          "applied_inputs": [], "existing_result": detail}
                save(directory / "hard-interruption.json", result)
            else:
                result = {"status": "existing_complete", "applied_inputs": [],
                          "stop_reason": "已有任务执行完成。" if detail.get("exit_code") == 0
                          else "已有任务返回失败，未自动修改代码。", "existing_result": detail}
            save(record_path, {"status": "completed", "arguments": arguments, "result": result})
            return result
    finally:
        history.close()


def run_work(job, emit, controls, *, model=None, environment_factory=TaskDocker, root=None):
    data = _job_data(job)
    root = Path(root or data.get("state_root") or state_root()).resolve()
    task_id = data["task_id"]
    if len(task_id) != 32 or any(c not in "0123456789abcdef" for c in task_id):
        raise ValueError("Invalid work identity")
    directory = root / "tasks" / task_id
    directory.mkdir(parents=True, exist_ok=True)
    owner = {"runner": None, "pause": None, "interrupted_operation": False}
    key = (str(root), task_id)
    with _owners_lock:
        if key in _owners:
            raise RuntimeError("Task already has an execution owner")
        _owners[key] = owner
    from agent.storage.resources import container_owner

    environment = None
    result = None
    owned_model = None
    runner = None
    # 旧检查点的环境/恢复失败不能降级成一次新的准备失败。
    fresh = not any((directory / name).exists() for name in (
        "history.sqlite3", "state.json", "existing/state.json",
    ))
    preparing = True
    try:
        failed_path = directory / "preparation-failed.json"
        recovery_id = next((item["id"] for item in reversed(controls())
                            if item["kind"] == "resume"), None)
        if failed_path.exists():
            if not recovery_id:
                return json.loads(failed_path.read_text(encoding="utf-8"))
            from agent.storage.records import immutable
            import hashlib

            raw = failed_path.read_bytes()
            immutable(directory / "preparation-failures" / (hashlib.sha256(raw).hexdigest() + ".json"), raw)
            failed_path.unlink()
        if (directory / "hard-interruption.json").exists() and not recovery_id:
            return {"status": "execution_unknown", "applied_inputs": [],
                    "stop_reason": "上次操作被强制中止；请先核对保留现场。"}
        if not fresh and data.get('kind') != 'existing':
            from agent.storage.history import LocalHistory
            from agent.storage.recovery import code_key

            archive = LocalHistory(directory)
            try:
                with archive.lease():
                    saved = archive.load()
                stop = (saved or {}).get('continuation') or {}
                requested_stop = (data.get('continuation') or {}).get('stop_id')
                if (recovery_id and requested_stop and stop.get('stop_id')
                        and requested_stop != stop['stop_id']):
                    raise ValueError('Task stop changed before dispatch; refresh before continuing')
                if (saved and saved.get('code_key') == code_key()
                        and saved.get('status') == 'verified_success' and not recovery_id):
                    from agent.storage.records import digest

                    identity = digest([
                        str(Path(data['repo']).resolve()), data['request'],
                        data.get('acceptance'), data.get('verify_command'),
                        data.get('public_conversation', []),
                        data.get('confirmed_requirements') or saved.get('initial_requirements'),
                    ])
                    if saved['identity'] != identity:
                        raise ValueError('Task requirements changed; saved success was not reused')
                    # 无人工继续时只补发已保存终态，不派发新的工作。
                    return {'status': 'verified_success', 'stop_reason': saved.get('reason', ''),
                            'usage': summarize(saved), 'result_reused': True,
                            'verification': saved.get('verification', {}),
                            'applied_inputs': [*saved.get('applied_inputs', []),
                                               *([recovery_id] if recovery_id else [])]}
            finally:
                archive.close()
        with control.active(root, task_id), container_owner("task:" + task_id, root=root):
            signal = control.current()

            def cancelled():
                return bool(signal.kind or owner["pause"] or any(
                    item["kind"] == "pause" for item in controls()
                ))

            if cancelled():
                result = {"status": "paused", "pause_requested": True, "applied_inputs": [
                    item["id"] for item in controls() if item["kind"] == "pause"
                ]}
                return result
            if data.get("project_location"):
                ensure_location(data["project_location"], directory)
            scope_path = directory / "project-scope.json"
            if scope_path.exists():
                scope = ProjectScope.from_record(json.loads(scope_path.read_text("utf-8")),
                                                 root=data["repo"])
                scope.validate(full_check=True)
            else:
                scope = ProjectScope.discover(data["repo"])
                save(scope_path, scope.to_record())
            if cancelled():
                return {"status": "paused", "pause_requested": True, "applied_inputs": []}
            binding_path = directory / "execution-binding.json"
            if binding_path.exists():
                binding = json.loads(binding_path.read_text(encoding="utf-8"))
            else:
                binding = data.get("execution_binding") or {}
                if not binding.get("image"):
                    from agent.execution.preparation import PreparationCancelled
                    try:
                        binding = prepare_binding(
                            data["repo"], cancelled=cancelled, project_scope=scope,
                            progress=lambda value: emit("environment", value),
                            **({"retry": True} if recovery_id else {}),
                        )
                    except PreparationCancelled:
                        # 尚未开始代码执行。暂停准备不应被误报成模型请求结果未知。
                        result = {
                            "status": "paused", "pause_requested": True,
                            "applied_inputs": [item["id"] for item in controls()
                                               if item["kind"] == "pause"],
                            "stop_reason": "环境准备等待已中止，未开始执行项目代码。",
                        }
                        save(directory / "preparation-interrupted.json", result)
                        return result
                binding = require_binding(data["repo"], binding)
                save(binding_path, binding)
            if cancelled():
                return {"status": "paused", "pause_requested": True, "applied_inputs": [
                    item["id"] for item in controls() if item["kind"] == "pause"
                ]}
            environment = environment_factory(
                data["repo"], binding["image"], directory,
                runtime=binding["runtime"], read_sources=data.get("read_sources", []),
                project_scope=scope,
                **({"managed": binding["managed"]} if binding.get("managed") else {}),
            )
            receipt_path = directory / "pause-container.json"
            if receipt_path.exists() and hasattr(environment, "restart_after_output_limit"):
                environment.restart_after_output_limit(
                    json.loads(receipt_path.read_text(encoding="utf-8"))
                )
                receipt_path.unlink()
            elif not fresh and recovery_id and hasattr(environment, "recover_stopped"):
                environment.recover_stopped(directory / "crash-container.json")
            else:
                environment.start()
            if data.get("kind") == "existing":
                preparing = False
                result = _existing(data, environment, directory, owner, emit, recovery_id)
                return result
            settings = data.get("settings", data.get("runtime_settings", {}))
            if model is None:
                owned_model = create_model(load_settings(settings))
            runner = WorkRunner(
                model or owned_model, environment,
                state_dir=directory, interactive=True, yield_boundary=False,
                event_sink=lambda event: None, emit=emit, control_source=controls,
                runtime_settings=settings, work_rules_factory=coding_rules,
                user_request_id=data.get("input_id", data["task_id"]),
                pause_after_steps=data.get("pause_after_steps"),
                confirmed_requirements=data.get("confirmed_requirements"),
                recovery_id=recovery_id,
            )
            owner["runner"] = runner
            history = messages_from_dict(data.get("public_conversation", []))
            result = runner.run(
                history, data["request"], StepBudget(**data.get("budget", {})),
                verify_command=data.get("verify_command"), acceptance=data.get("acceptance"),
            )
            preparing = False
            result["applied_inputs"] = list(runner.state.get("applied_inputs", []))
            if owner["interrupted_operation"]:
                result.update(
                    status="execution_unknown", pause_requested=True,
                    stop_reason="用户已立即中止正在执行的操作；部分改动可能已发生，请先核对现场。",
                )
                # 独立中止记录在恢复入口先核验，不能被随后收尾的旧检查点覆盖。
                save(directory / "hard-interruption.json", {
                    "command_id": owner["pause"], "status": "execution_unknown",
                    "reason": result["stop_reason"],
                })
                from agent.storage.history import LocalHistory
                archive = LocalHistory(directory)
                try:
                    with archive.lease():
                        stopped = archive.load()
                        stopped.update(status="execution_unknown", hard_interruption={
                            "command_id": owner["pause"], "reason": result["stop_reason"],
                        })
                        archive.record(stopped, "hard_interrupted", stopped["hard_interruption"])
                finally:
                    archive.close()
            result.pop("messages", None)
            emit("work_state", {
                "status": result["status"], "reason": result.get("stop_reason", ""),
                "task_id": task_id, "usage": result.get("usage", {}),
                "applied_inputs": result["applied_inputs"],
                "acceptance_limit": result.get("acceptance_limit"),
                "continuation": result.get("continuation"),
            })
            return result
    except Exception as exc:
        # 只在首次执行前归类准备失败。已有工作状态、模型请求及实际操作的
        # 记录故障仍交给上层按关键故障处理，不能伪称只是复制未完成。
        if not fresh or not preparing or (runner is not None and runner.state):
            raise
        from agent.execution.projects import EnvironmentRequired
        from agent.source.artifacts import ArtifactScopeError
        from agent.source.snapshots import SnapshotError

        if isinstance(exc, OSError) and (
            exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) == 112
            or str(exc).startswith("磁盘空间不足：")
        ):
            detail = "保存环境或源码时磁盘空间不足。"
        elif isinstance(exc, (EnvironmentRequired, SnapshotError, ArtifactScopeError,
                              ProjectLocationError, ProjectScopeError)):
            detail = str(exc)[:1000]
        elif isinstance(exc, InterruptedError):
            detail = "环境或源码准备已中止。"
        else:
            detail = "环境或源码准备失败（" + type(exc).__name__ + "）。"
        result = {
            "status": "preparation_failed", "task_id": task_id, "applied_inputs": [],
            "stop_reason": detail + " 已停止，未开始项目代码执行；原现场保留，不会自动重试。",
            "error_type": type(exc).__name__,
        }
        # 若连这份小记录也无法提交，继续抛出存储故障；不能宣称已保存。
        save(directory / "preparation-failed.json", result)
        return result
    finally:
        try:
            if environment is not None:
                environment.close()
                if (not owner["interrupted_operation"]
                        and hasattr(environment, "output_limit_stop_receipt")):
                    save(directory / "pause-container.json", environment.output_limit_stop_receipt())
            # 停止只关闭执行者。任务资源引用由明确永久删除入口释放。
        except Exception as exc:
            # 已保存的实际结果不能被清理失败改写；保留可重试清理的诊断。
            save(directory / "resource-cleanup-error.json", {
                "error_type": type(exc).__name__, "previous_result": result,
            })
            if result is not None:
                result.update(status="execution_unknown", stop_reason="停止尚未核验，失联租约已撤销；请核对容器状态后继续。")
            save(directory / "hard-interruption.json", {
                "status": "execution_unknown", "reason": "Container stop was not confirmed",
            })
            emit("cleanup_pending", {"task_id": task_id, "reason": "资源收尾未完成，现场仍保留。"})
        finally:
            with _owners_lock:
                _owners.pop(key, None)
            if owned_model is not None:
                from agent.model.transport import close_model
                try:
                    close_model(owned_model)
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("Work model cleanup failed")
