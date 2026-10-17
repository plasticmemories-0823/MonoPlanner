"""执行已保留的操作，并记录真实验收结果。"""

from langchain_core.messages import ToolMessage, message_to_dict
import json
from agent.source import lifecycle as source_runtime
from agent.source.snapshots import SourceSnapshot
from agent.source.errors import SourceStorageError
from agent.task.contracts import WorkRequest
from agent.acceptance.contracts import VerificationSpec
from agent.acceptance.final import verify_final
from agent.acceptance import revisions
from agent.storage.records import digest, save
from agent.model.errors import ModelInterrupted, ModelRequestNotSent
from agent.rag.rewrite import RewriteUnknown


def execute_pending(self):
    pending = self.state["pending"]
    source_runtime.collect(self)
    with self.executor.probe_batch(
        self.root / "plans" / pending["id"] / "probe-sessions",
        enabled=pending["plan"]["kind"] == "inspect",
    ):
        return self._execute_pending()


def _execute_pending(self):
    pending = self.state["pending"]
    plan = WorkRequest.model_validate(pending["plan"])
    if pending["rows"] and pending["rows"][-1]["status"] == "failed":
        return self.finish_plan("execution_failed")
    if self.state["rounds"] and self.state["rounds"][-1]["plan"] == pending["id"]:
        return self.finish_plan(
            "verified_success"
            if self.state["verification"]["status"] == "passed"
            else "verification_failed"
        )
    if plan.kind == "candidate" and len(self.state["rounds"]) >= self.max_rounds:
        return self.finish_plan("candidate_limit_reached")
    prior = {r["id"]: r for r in pending["rows"]}
    while pending["cursor"] < len(plan.steps):
        self.budget.check_request(0)
        index = pending["cursor"]
        step = plan.steps[index]
        directory = self.root / "plans" / pending["id"] / (pending["id"] + "-" + step.id)
        directory.mkdir(parents=True, exist_ok=True)
        read_only = source_runtime.readonly(self, plan, step)
        if read_only:
            before = self.evidence.record_reference(
                'step_before', self.evidence.current, plan=pending['id'], step=step.id
            )
        elif self.optimize_source:
            before = source_runtime.boundary(self, 'step_before', plan=pending['id'], step=step.id)
        else:
            before = self.evidence.record('step_before', plan=pending['id'], step=step.id)
        self.state["running_step"] = {
            "plan": pending["id"],
            "id": step.id,
            "before": before,
            "op": step.op,
        }
        save(directory / "request.json", step.model_dump())
        self.persist("step_started", self.state["running_step"])
        source_runtime.configure_lookups(self)
        timeout = min(
            self.environment.timeout, self.budget.remaining_time() or self.environment.timeout
        )
        cache_key = (
            digest([before, self.env_key, step.model_dump(exclude={"id"})])
            if step.op in {"read", "search"}
            and not step.from_step
            and not step.path.startswith("$")
            else None
        )
        if step.op == 'read' and self.executor.has_visible_overlap(step):
            cache_key = None  # 依赖上下文的摘录不能替换完整原始缓存。
        try:
            cached = (
                self.history_store.cached(cache_key) if self.history_store and cache_key else None
            )
            row = (
                {"status": cached["status"], "data": cached["data"], "seconds": 0}
                if cached
                else self.executor.execute(
                    step, prior, directory, timeout, isolated=plan.kind == "inspect"
                )
            )
            source_runtime.record_lookup(self, step, row, directory)
        except (ModelInterrupted, RewriteUnknown):
            # 未知检索改写和执行中断保留 running_step，不能交回模型后自动续跑。
            raise
        except SourceStorageError:
            # 宿主源码副本/证据写入失败必须停止，不能作为项目失败交回模型再复制。
            raise
        except Exception as exc:
            row = {"status": "failed", "data": {"error": f"{type(exc).__name__}: {exc}"}}
        after = (
            self.evidence.record_reference('step_after', before, plan=pending['id'], step=step.id)
            if read_only
            else self.evidence.record('step_after', plan=pending['id'], step=step.id)
        )
        row.update(
            cache_key=cache_key,
            id=step.id,
            op=step.op,
            before=before,
            after=after,
            record=(directory / "result.json").relative_to(self.root).as_posix(),
        )
        save(directory / "result.json", row)
        pending["rows"].append(row)
        pending["cursor"] += 1
        self.state.update(running_step=None, source=after)
        prior[step.id] = row
        self.steps_this_invocation += 1
        self.persist("step_finished", row)
        if row["status"] == "failed":
            return self.finish_plan("execution_failed")
        if self.yield_boundary:
            return
        # 只有所属执行者在操作完成后的边界检查排队输入。
        # 会话消息必须先经过聊天分类，才能改变工作。
        if getattr(self, 'mailbox', None):
            applied = set(self.state.get('applied_inputs', []))
            if any(r['id'] not in applied for r in self.mailbox()):
                return
        elif self.control_source and any(
            r['id'] not in self.state.get('applied_inputs', []) for r in self.control_source()
        ):
            return
        if self.pause_after_steps and self.steps_this_invocation >= self.pause_after_steps:
            self.state.update(
                status="paused", reason="Requested pause at a completed step boundary"
            )
            self.persist("paused")
            return
    if plan.kind == "candidate":
        return self.final_acceptance()
    return self.finish_plan("evidence_ready")


def final_acceptance(self):
    if len(self.state["rounds"]) >= self.max_rounds:
        return self.finish_plan("candidate_limit_reached")
    pending = self.state["pending"]
    folder = self.root / "feedback" / pending["id"]
    if self.optimize_source:
        source_runtime.boundary(self, "candidate_boundary", plan=pending["id"])
    snapshot = SourceSnapshot.capture(
        self.environment.repo_root,
        folder / "candidate",
        runtime_files=getattr(self.environment, "runtime_files", ()),
        project_scope=getattr(self.environment, "project_scope", None),
    )
    if self.optimize_source and snapshot.files != dict(self.evidence.view().files):
        raise ModelRequestNotSent('state_conflict', 'Source changed while capturing the candidate.')
    self.state['verification_running'] = True
    self.persist("final_acceptance_started", {"plan": pending["id"], "source": snapshot.digest})
    timeout = min(
        self.environment.timeout, self.budget.remaining_time() or self.environment.timeout
    )
    self.fresh_verification = True
    report = verify_final(
        self.environment,
        snapshot,
        VerificationSpec(**self.state["verification_spec"]),
        self.state["regression_command"],
        folder / "verification",
        timeout,
        repository_baseline=(
            revisions.checked(
                self.root,
                revisions.checked(self.root, self.state["active_acceptance"])["baseline_result"],
            )
            if self.state.get("active_acceptance")
            else None
        ),
    )
    report = revisions.compare(self, snapshot, report, folder / "acceptance-versions")
    save(folder / "result.json", report)
    snapshot.write_patch(self.baseline, self.root / "candidate.patch")
    self.state["rounds"].append(
        {"plan": pending["id"], "verification": report, "candidate_digest": snapshot.digest}
    )
    self.state.update(verification=report, verification_running=False)
    self.persist(
        "final_acceptance_finished",
        {"plan": pending["id"], "record": str(folder / "result.json"), "status": report["status"]},
    )
    self.finish_plan("verified_success" if report["status"] == "passed" else "verification_failed")


def finish_plan(self, outcome):
    pending = self.state["pending"]
    result = {
        "outcome": outcome,
        "plan": pending["id"],
        "steps": [{k: v for k, v in r.items() if k != "cache_key"} for r in pending["rows"]],
    }
    if outcome == "candidate_limit_reached":
        result["final_acceptance"] = {
            "status": "not_run",
            "reason": "Final candidate attempt limit reached",
        }
    if outcome.startswith("verification") or outcome == "verified_success":
        report = self.state["verification"]
        result["verification"] = {
            k: report.get(k)
            for k in (
                "status",
                "reason",
                "stderr",
                "test_results",
                "regression",
                "tests_ready",
                "execution_exception",
                "acceptance_comparison",
                "test_groups",
                "repository_skips",
                "repository_baseline_matched",
            )
        }
        if self.state.get("active_acceptance"):
            result["acceptance_trial_id"] = self.state["active_acceptance"]["id"]
    save(self.root / "plans" / pending["id"] / "result.json", result)
    from agent.model.tool_payloads import project
    from agent.model.tool_results import normalize_result

    normalized = normalize_result(
        "submit_plan", result, source_version=self.state.get("source"),
        project=getattr(getattr(self, "environment", None), "repo_root", None),
    )
    delivered = project(self, normalized)
    self.state["history"].extend(
        [
            pending["response"],
            message_to_dict(
                ToolMessage(
                    content=json.dumps(delivered, ensure_ascii=False),
                    name=pending["response"]["data"]["tool_calls"][0]["name"],
                    tool_call_id=pending["response"]["data"]["tool_calls"][0]["id"],
                )
            ),
        ]
    )
    status = "verified_success" if outcome == "verified_success" else "needs_plan"
    reason = "Fixed final acceptance passed" if outcome == "verified_success" else outcome
    if outcome == "candidate_limit_reached":
        status, reason = "stalled", result["final_acceptance"]["reason"]
    self.state.update(
        last_plan=pending["id"],
        last_result=result,
        pending=None,
        status=status,
        reason=reason,
    )
    self.persist("plan_finished", {"plan": self.state["last_plan"], "outcome": outcome})
