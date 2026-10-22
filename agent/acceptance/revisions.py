"""在原始源码上试跑生成的验收，保留每个审查版本。"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Literal

from langchain_core.messages import ToolMessage, message_to_dict
from pydantic import BaseModel, ConfigDict, Field

from agent.acceptance.contracts import VerificationSpec, tests_ready_for_candidate
from agent.acceptance import corrections, evidence
from agent.acceptance.execution import ExecutionCheck, checked_execution
from agent.acceptance.evidence import Correction, Evidence
from agent.acceptance.runner import AcceptanceRun
from agent.execution.local import LocalEnvironment
from agent.model.errors import ModelRequestNotSent
from agent.source.snapshots import SourceSnapshot, _safe_file
from agent.storage.records import digest, save


# ============================================================
# 审查结果与交付原件
# ============================================================
class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["freeze", "revise"]
    trial_id: str
    reason: str = Field(min_length=1, max_length=1600)
    test_code: str = Field(default="", max_length=32768)
    test_driver: list[str] = Field(default_factory=list, max_length=32)
    execution_checks: list[ExecutionCheck] = Field(default_factory=list, max_length=8)
    regression_command: str = Field(default="", max_length=16000)
    evidence: list[Evidence] = Field(default_factory=list, max_length=24)
    corrections: list[Correction] = Field(default_factory=list, max_length=24)


def checked(root, ref):
    raw = _safe_file(root, Path(ref["record"])).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ModelRequestNotSent("state_conflict", "Acceptance record changed: " + ref["record"])
    return json.loads(raw)


def reference(root, path):
    return {
        "record": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def candidate_record(loop, plan, response):
    data = {"plan": plan.model_dump(), "response": response, "source": loop.evidence.current}
    data["work_sha256"] = digest(data["plan"]["steps"])
    path = loop.root / "acceptance" / "candidates" / (digest(data) + ".json")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != data:
            raise ModelRequestNotSent("state_conflict", "Pending candidate record changed")
    else:
        save(path, data)
    return reference(loop.root, path)


def candidate_summary(loop, ref):
    if not ref:
        return None
    data = checked(loop.root, ref)
    return {
        **ref,
        "record": "$record/" + ref["record"],
        "source": data["source"],
        "work_sha256": data["work_sha256"],
        "call_id": data["response"]["data"]["tool_calls"][0]["id"],
    }


def released_plan(candidate, proposal):
    plan = deepcopy(candidate["plan"])
    plan.update(
        test_code=proposal["spec"]["test_code"],
        regression_command=proposal["regression"],
        test_driver=list(proposal["spec"].get("test_driver", ())),
        execution_checks=list(proposal["spec"].get("execution_checks", ())),
    )
    return plan


# ============================================================
# 原始运行条件与结果记录
# ============================================================
def baseline_conditions(loop, spec, regression):
    from agent.storage.recovery import environment_key

    return digest(
        {
            "environment": environment_key(loop.environment),
            "timeout": loop.environment.timeout,
            "original_source": loop.baseline.digest,
            "workspace_source": loop.evidence.current,
            "spec": asdict(spec),
            "regression": regression,
        }
    )


def append_result(loop, response, result):
    call = response["data"]["tool_calls"][0]
    from agent.model.tool_payloads import project
    from agent.model.tool_results import normalize_result

    normalized = normalize_result(
        call["name"], result, source_version=loop.state.get("source"),
        project=getattr(getattr(loop, "environment", None), "repo_root", None),
    )
    delivered = project(loop, normalized)
    loop.state["history"].extend(
        [
            response,
            message_to_dict(
                ToolMessage(
                    content=json.dumps(delivered, ensure_ascii=False),
                    name=call["name"],
                    tool_call_id=call["id"],
                )
            ),
        ]
    )
    loop.state.update(response_pending=None, plan_error="", repair_count=0)


def spec_for(loop, code, driver=(), execution_checks=()):
    request = loop.state["request"] + ''.join(
        '\n' + row['text'] for row in loop.state.get('user_updates', [])
    )
    criteria = tuple([request] + ([loop.state["acceptance"]] if loop.state["acceptance"] else []))
    from agent.task.requirements import check_execution
    from agent.task.repair import ValidationErrors, fields

    requirement_digest = ""
    errors, checks = [], ()
    value = {"test_code": code, "test_driver": driver, "execution_checks": execution_checks}
    try:
        checks = checked_execution(execution_checks, criteria, required=True)
    except ValueError as error:
        errors.append((["/execution_checks"], ValueError(str(error))))
    else:
        try:
            requirement_digest = check_execution(loop, checks)
        except ValueError as error:
            errors.append((fields(error, value), error))
    try:
        spec = VerificationSpec(
            criteria, test_code=code, test_driver=driver, execution_checks=checks,
            requirement_digest=requirement_digest,
        )
    except ValueError as error:
        errors.append((fields(error, value), error))
    if errors:
        raise ValidationErrors(errors)
    return spec


def run_report(loop, snapshot, spec, directory, phase, repository_baseline=None):
    # 本地执行没有系统级沙箱；必须使用一次性副本，不操作保留的原件。
    if isinstance(loop.environment, LocalEnvironment):
        snapshot = SourceSnapshot.capture(snapshot.root, directory / "source")
    timeout = min(
        loop.environment.timeout, loop.budget.remaining_time() or loop.environment.timeout
    )
    with snapshot.environment_like(
        loop.environment, directory / "runtime", read_only=True
    ) as backend:
        report = AcceptanceRun(
            snapshot.root,
            spec,
            artifacts_root=directory / "runtime",
            environment=backend,
            timeout=timeout,
            repository_baseline=repository_baseline,
        ).run(phase)
        integrity = snapshot.verify_integrity()
    report["source_integrity"] = integrity
    if not integrity["valid"]:
        report.update(
            status="error",
            tests_ready=False,
            integrity_ok=False,
            error_kind="integrity_error",
            reason="Acceptance changed its isolated source copy",
        )
    return report


# ============================================================
# 验收记录的完整性校验
# ============================================================
def validate(loop):
    from agent.task.requirements import validate_state

    validate_state(loop)
    if loop.state.get("acceptance_trial") or (
        loop.state.get("verification_spec") or {}
    ).get("test_code"):
        from agent.acceptance.boundaries import require_general

        require_general(loop)
    checked_proposals = set()

    def proposal(ref):
        data = checked(loop.root, ref)
        if data.get("original_source") != loop.baseline.digest:
            raise ModelRequestNotSent("state_conflict", "Acceptance original snapshot identity changed")
        key = (ref["record"], ref["sha256"])
        if key in checked_proposals:
            return data
        if data.get("candidate"):
            checked(loop.root, data["candidate"])
        correction = data.get("acceptance_correction")
        if correction:
            try:
                base_ref = correction["base_version"]
                if base_ref not in loop.state.get("acceptance_versions", []):
                    raise ValueError("Correction base is not a retained frozen version")
                base = checked(loop.root, base_ref)
                expected = evidence.prepare(
                    loop, base_ref, base["spec"], data["spec"],
                    [evidence.Evidence(**{key: row[key] for key in ("id", "source", "quote")})
                     for row in correction["evidence"]],
                    [evidence.Correction(**row) for row in correction["corrections"]],
                    base["regression"], data["regression"],
                )
                verified = spec_for(loop, data["spec"]["test_code"],
                                    data["spec"].get("test_driver", ()),
                                    data["spec"].get("execution_checks", ()))
                if (expected != correction
                        or json.loads(json.dumps(asdict(verified))) != data["spec"]):
                    raise ValueError("Correction differs from its original evidence or fixed checks")
            except (ValueError, KeyError, TypeError) as exc:
                raise ModelRequestNotSent(
                    "state_conflict", "Acceptance correction changed: " + str(exc)
                ) from exc
        checked_proposals.add(key)
        return data

    for ref in loop.state.get("acceptance_trials", []):
        data = checked(loop.root, ref)
        if data.get("proposal"):
            proposal(data["proposal"])
        elif data.get("candidate"):
            checked(loop.root, data["candidate"])
        if data.get("result"):
            checked(loop.root, data["result"])
    for ref in loop.state.get("acceptance_versions", []):
        data = proposal(ref)
        if data.get("baseline_result"):
            checked(loop.root, data["baseline_result"])
    trial = loop.state.get("acceptance_trial")
    if trial:
        data = proposal(trial["ref"])
        if trial.get("candidate") != data.get("candidate"):
            raise ModelRequestNotSent("state_conflict", "Pending candidate selection changed")
        if trial.get("baseline_result"):
            checked(loop.root, trial["baseline_result"])
    release = loop.state.get("acceptance_release")
    if release:
        approval = proposal(release["approval_ref"])
        candidate = checked(loop.root, release["candidate_ref"])
        if (
            release["approval_ref"] != loop.state.get("active_acceptance")
            or release["candidate_ref"] != approval.get("candidate")
            or release["plan"] != released_plan(candidate, approval)
            or release["origin_response"] != candidate["response"]
            or release["response"] != approval["review_response"]
        ):
            raise ModelRequestNotSent(
                "state_conflict", "Released candidate differs from its reviewed original"
            )


# ============================================================
# 原始源码上的验收试跑
# ============================================================
def trial(
    loop,
    code,
    regression,
    response,
    *,
    driver=(),
    execution_checks=(),
    candidate=None,
    parent=None,
    reason="",
    reuse=None,
    superseded=None,
    acceptance_correction=None,
):
    spec = spec_for(loop, code, driver, execution_checks)
    proposal = {
        "spec": asdict(spec),
        "regression": regression,
        "parent": parent,
        "reason": reason,
        "original_source": loop.baseline.digest,
        "workspace_source": loop.evidence.current,
        "candidate": candidate,
        "baseline_conditions": baseline_conditions(loop, spec, regression),
    }
    if acceptance_correction:
        proposal["acceptance_correction"] = acceptance_correction
    identifier = hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()
    folder = loop.root / "acceptance" / identifier
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "proposal.json"
    if path.exists():
        if (digest(json.loads(_safe_file(loop.root, path.relative_to(loop.root)).read_bytes())) != digest(proposal)
                or any(ref.get('id') == identifier for ref in loop.state.get('acceptance_trials', []))):
            raise ValueError("Identical acceptance proposal already trialled; use its trial_id")
    save(path, proposal)
    ref = {**reference(loop.root, path), "id": identifier}
    loop.state.setdefault("acceptance_trials", []).append(ref)
    loop.state.update(
        acceptance_trial={"ref": ref, "candidate": candidate},
        acceptance_running=True,
        response_pending=None,
    )
    # 草稿已接收为待试跑验收；与试跑检查点一起消费，不能再次用于修正旧提交。
    loop.state.pop("repair_draft", None)
    save(folder / "continuation.json", {
        "response": response, "candidate": candidate, "superseded": superseded,
        "reuse": reuse, "acceptance_correction": acceptance_correction,
    })
    loop.state["acceptance_trial"]["continuation"] = reference(loop.root, folder / "continuation.json")
    loop.persist("acceptance_trial_started", {"trial_id": identifier})
    report = (
        checked(loop.root, reuse)
        if reuse
        else run_report(loop, loop.baseline, spec, folder / "baseline", "baseline")
    )
    finish_trial(loop, folder, ref, report, response, candidate, superseded, reuse,
                 acceptance_correction)


def resume_trial(loop, *, allow_execute):
    current = loop.state["acceptance_trial"]
    ref = current["ref"]
    proposal = checked(loop.root, ref)
    folder = loop.root / "acceptance" / ref["id"]
    original = checked(loop.root, current['continuation'])
    result = folder / "baseline-result.json"
    if original['reuse']:
        report = checked(loop.root, original['reuse'])
    elif result.exists():
        report = json.loads(_safe_file(loop.root, result.relative_to(loop.root)).read_bytes())
        completed = json.loads(_safe_file(loop.root, folder.relative_to(loop.root) / "completion.json").read_bytes())
        if completed != {"result_digest": digest(report), "proposal": ref}:
            raise ValueError("Saved acceptance trial result checksum mismatch")
    elif allow_execute:
        import uuid

        report = run_report(loop, loop.baseline, VerificationSpec(**proposal['spec']),
                            folder / ('baseline-recovery-' + uuid.uuid4().hex), 'baseline')
    else:
        return
    finish_trial(loop, folder, ref, report, original['response'], original['candidate'],
                 original['superseded'], original['reuse'], original['acceptance_correction'])


def finish_trial(loop, folder, ref, report, response, candidate, superseded, reuse,
                 acceptance_correction):
    identifier = ref["id"]
    if not loop.baseline.verify_integrity()["valid"]:
        raise ModelRequestNotSent(
            "state_conflict", "Retained original source changed during acceptance trial"
        )
    if reuse:
        result_ref = reuse
    else:
        result_path = folder / "baseline-result.json"
        save(folder / "completion.json", {"result_digest": digest(report), "proposal": ref})
        save(result_path, report)
        result_ref = reference(loop.root, result_path)
    completion = folder / "trial.json"
    save(completion, {"proposal": ref, "result": result_ref})
    loop.state["acceptance_trials"][-1] = reference(loop.root, completion)
    loop.state["acceptance_trial"].update(baseline_result=result_ref)
    loop.state["acceptance_running"] = False
    append_result(
        loop,
        response,
        {
            "outcome": "acceptance_preflight",
            "trial_id": identifier,
            "source": "original, before task edits",
            "workspace_modified": False,
            "baseline": report,
            "proposal_record": "$record/" + ref["record"],
            "result_record": "$record/" + result_ref["record"],
            "selected_candidate": candidate_summary(loop, candidate),
            "superseded_candidate": candidate_summary(loop, superseded),
            "baseline_reused": bool(reuse),
            "acceptance_correction": acceptance_correction,
            "next": "Inspect the real evidence, then freeze_acceptance or revise_acceptance. "
            "A runnable assertion failure may reproduce the defect; it is not proof that the expectation is correct. "
            "No proposed modification has run. freeze_acceptance releases only the selected_candidate above. "
            "revise_acceptance changes tests only and keeps that candidate; to replace its edits before the first "
            "freeze, submit a complete candidate with submit_plan and review the new trial_id.",
        },
    )
    loop.persist("acceptance_trial_finished", {"trial_id": identifier, "status": report["status"]})


# ============================================================
# 保存等待冻结的候选修改
# ============================================================
def stage_candidate(loop, plan, response):
    from agent.acceptance.boundaries import require_general

    require_general(loop)
    if loop.state.get("verification_spec") or loop.state.get("verify_command"):
        raise ValueError("Pending replacement cannot change frozen acceptance")
    validate(loop)
    current = loop.state.get("acceptance_trial")
    reuse = None
    if current:
        if loop.state.get("acceptance_running") or not current.get("baseline_result"):
            raise ModelRequestNotSent(
                "execution_unknown", "Pending acceptance trial has no completed result"
            )
        proposal = checked(loop.root, current["ref"])
        if proposal["workspace_source"] != loop.evidence.current:
            raise ModelRequestNotSent(
                "state_conflict", "Workspace changed while acceptance was under review"
            )
        report = checked(loop.root, current["baseline_result"])
        conditions = baseline_conditions(
            loop, spec_for(loop, plan.test_code, plan.test_driver, plan.execution_checks),
            plan.regression_command,
        )
        remaining = loop.budget.remaining_time()
        if (
            proposal["baseline_conditions"] == conditions
            and tests_ready_for_candidate(report)
            and (remaining is None or remaining >= loop.environment.timeout)
        ):
            reuse = current["baseline_result"]
    trial(
        loop,
        plan.test_code,
        plan.regression_command,
        response,
        driver=plan.test_driver,
        execution_checks=plan.execution_checks,
        candidate=candidate_record(loop, plan, response),
        reuse=reuse,
        parent=current["ref"]["id"] if current else None,
        superseded=current.get("candidate") if current else None,
        reason="Explicit pending candidate replacement" if current else "",
    )


# ============================================================
# 冻结与受保护的验收修订
# ============================================================
def review(loop, decision, response):
    validate(loop)
    current_trial = loop.state.get("acceptance_trial")
    current_ref = current_trial["ref"] if current_trial else loop.state.get("active_acceptance")
    if not current_ref or decision.trial_id != current_ref["id"]:
        raise ValueError("Review requires the current acceptance trial_id")
    proposal = checked(loop.root, current_ref)
    if proposal["workspace_source"] != loop.evidence.current and current_trial:
        raise ModelRequestNotSent(
            "state_conflict", "Workspace changed while acceptance was under review"
        )
    if decision.decision == "revise":
        if loop.state.get("verify_command"):
            raise ValueError("User fixed acceptance is an original requirement and cannot be revised. "
                             + evidence.REVISION_GUIDANCE)
        if not decision.test_code.strip():
            raise ValueError("Revision requires complete test_code; no code is guessed")
        fields = {item.field for item in decision.corrections}
        driver = (decision.test_driver if decision.test_driver or "test_driver" in fields
                  else proposal["spec"].get("test_driver", ()))
        execution_checks = (
            [item.model_dump() for item in decision.execution_checks]
            if decision.execution_checks or "execution_checks" in fields
            else proposal["spec"].get("execution_checks", ())
        )
        regression = (decision.regression_command
                      if decision.regression_command or "regression_command" in fields
                      else proposal["regression"])
        frozen = loop.state.get("verification_spec")
        correction = None
        if frozen:
            if not decision.evidence or not decision.corrections:
                raise ValueError(evidence.REVISION_GUIDANCE)
            base_ref = loop.state["active_acceptance"]
            base = checked(loop.root, base_ref)
            try:
                spec = spec_for(loop, decision.test_code, driver, execution_checks)
                correction = evidence.prepare(
                    loop, base_ref, base["spec"], asdict(spec),
                    decision.evidence, decision.corrections, base["regression"], regression,
                )
            except ValueError as error:
                if evidence.REVISION_GUIDANCE in str(error):
                    raise
                raise ValueError(str(error) + ". " + evidence.REVISION_GUIDANCE) from error
        elif decision.evidence or decision.corrections:
            raise ValueError("Evidence corrections apply after freezing; revise the draft normally")
        trial(
            loop,
            decision.test_code,
            regression,
            response,
            driver=driver,
            execution_checks=execution_checks,
            candidate=current_trial.get("candidate") if current_trial else None,
            parent=current_ref["id"],
            reason=decision.reason,
            acceptance_correction=correction,
        )
        return
    if not current_trial or not current_trial.get("baseline_result"):
        raise ValueError("No completed acceptance trial can be frozen")
    if (decision.test_code or decision.test_driver or decision.execution_checks
            or decision.evidence or decision.corrections or decision.regression_command):
        raise ValueError(
            "freeze_acceptance confirms existing code; use revise_acceptance to submit code"
        )
    from agent.task.requirements import guard

    guard(loop, proposal["spec"])
    report = checked(loop.root, current_trial["baseline_result"])
    if not tests_ready_for_candidate(report):
        raise ValueError(
            "Acceptance is not runnable; inspect its baseline errors and revise it before freezing"
        )
    version = {
        **proposal,
        "baseline_result": current_trial["baseline_result"],
        "review_reason": decision.reason,
        "review_response": response,
    }
    version_path = loop.root / "acceptance" / current_ref["id"] / "frozen.json"
    save(version_path, version)
    ref = {**reference(loop.root, version_path), "id": current_ref["id"]}
    loop.state.setdefault("acceptance_versions", []).append(ref)
    loop.state.update(
        active_acceptance=ref,
        verification_spec=proposal["spec"],
        regression_command=proposal["regression"],
        acceptance_trial=None,
    )
    save(
        loop.root / "frozen-acceptance.json",
        {"spec": proposal["spec"], "regression": proposal["regression"]},
    )
    candidate_ref = current_trial.get("candidate")
    if candidate_ref:
        candidate = checked(loop.root, candidate_ref)
        loop.state["acceptance_release"] = {
            "plan": released_plan(candidate, proposal),
            "response": response,
            "origin_response": candidate["response"],
            "candidate_ref": candidate_ref,
            "approval_ref": ref,
        }
    else:
        append_result(
            loop,
            response,
            {
                "outcome": "acceptance_frozen",
                "trial_id": ref["id"],
                "acceptance_correction": proposal.get("acceptance_correction"),
                "next": "Submit source repair using this acceptance; previous versions remain recorded.",
            },
        )
    loop.persist("acceptance_frozen", {"trial_id": ref["id"]})


# ============================================================
# 验收条件的前后比较
# ============================================================
def compare(loop, snapshot, active_report, folder):
    validate(loop)
    versions = loop.state.get("acceptance_versions", [])
    records = [checked(loop.root, ref) for ref in versions]
    active = checked(loop.root, loop.state["active_acceptance"]) if versions else None
    rows, conflicts = [], []
    for index, (ref, version) in enumerate(zip(versions, records)):
        baseline = checked(loop.root, version["baseline_result"])
        report = (
            active_report
            if ref == loop.state.get("active_acceptance")
            else run_report(
                loop,
                snapshot,
                VerificationSpec(**version["spec"]),
                folder / ref["id"],
                "candidate",
                baseline,
            )
        )
        # 旧结果保留对照；只投射能精确绑定旧材料的已冻结修订，
        # 未声明的设置、断言和其它历史差异仍保留，不能被新版本整体遮盖。
        effective_spec = deepcopy(version["spec"])
        effective_regression = version["regression"]
        superseded_by = []
        for later_ref, later in zip(versions[index + 1:], records[index + 1:]):
            correction = later.get("acceptance_correction")
            if not correction:
                continue
            base = checked(loop.root, correction["base_version"])
            if corrections.protected(effective_spec) != corrections.protected(base["spec"]):
                continue
            try:
                material = corrections.project(
                    corrections.materials(effective_spec, effective_regression),
                    corrections.materials(base["spec"], base["regression"]),
                    [evidence.Correction(**row) for row in correction["corrections"]],
                )
            except ValueError:
                continue  # 旧片段身份不能确认时，仍保留原来的冲突检查。
            effective_regression = material.pop("regression_command")
            effective_spec.update(material)
            superseded_by.append(later_ref["id"])
        effective_report, effective_baseline = report, baseline
        if superseded_by:
            if effective_spec == active["spec"]:
                effective_report = active_report
                effective_baseline = checked(loop.root, active["baseline_result"])
            else:
                spec = VerificationSpec(**effective_spec)
                projected = folder / ref["id"] / "corrected-acceptance"
                effective_baseline = run_report(
                    loop, loop.baseline, spec, projected / "baseline", "baseline"
                )
                effective_report = run_report(
                    loop, snapshot, spec, projected / "candidate", "candidate", effective_baseline
                )
        if ref != loop.state.get("active_acceptance") and not (
            superseded_by and effective_spec == active["spec"]
        ) and (
            effective_report.get("status") != "passed"
            and (tests_ready_for_candidate(effective_baseline)
                 or tests_ready_for_candidate(effective_report))
        ):
            conflicts.append(ref["id"])
        rows.append({"trial_id": ref["id"], "baseline": baseline, "candidate": report,
                     "superseded_acceptance_by": superseded_by,
                     "effective_spec": effective_spec, "effective_baseline": effective_baseline,
                     "effective_regression": effective_regression,
                     "effective_candidate": effective_report})
    result = {
        "versions": rows,
        "conflicts": conflicts,
        "meaning": "Execution comparisons, not a proof of semantic coverage or correct expectations.",
    }
    save(folder / "comparison.json", result)
    active_report["acceptance_comparison"] = {
        "record": "$record/" + (folder / "comparison.json").relative_to(loop.root).as_posix(),
        "conflicts": conflicts,
    }
    if conflicts:
        active_report.update(
            status="failed", reason="Retained acceptance has unresolved runnable failures"
        )
    return active_report
