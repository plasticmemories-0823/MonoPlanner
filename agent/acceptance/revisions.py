"""Trial generated acceptance on original source; retain every reviewed version."""
import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Literal

from langchain_core.messages import ToolMessage, message_to_dict
from pydantic import BaseModel, ConfigDict, Field

from agent.acceptance.contracts import VerificationSpec, tests_ready_for_candidate
from agent.acceptance.runner import AcceptanceRun
from agent.acceptance.artifact_checks import required_consumer, requested_calls
from agent.execution.local import LocalEnvironment
from agent.model.errors import ModelRequestNotSent
from agent.source.snapshots import SourceSnapshot, _safe_file
from agent.storage.records import digest, save


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["freeze", "revise"]
    trial_id: str
    reason: str = Field(min_length=1, max_length=1600)
    test_code: str = Field(default="", max_length=32768)
    test_driver: list[str] = Field(default_factory=list, max_length=32)


def checked(root, ref):
    raw = _safe_file(root, Path(ref["record"])).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ModelRequestNotSent("state_conflict", "Acceptance record changed: " + ref["record"])
    return json.loads(raw)


def reference(root, path):
    return {"record": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


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
    return {**ref, "record": "$record/" + ref["record"], "source": data["source"],
            "work_sha256": data["work_sha256"],
            "call_id": data["response"]["data"]["tool_calls"][0]["id"]}


def released_plan(candidate, proposal):
    plan = deepcopy(candidate["plan"])
    plan.update(test_code=proposal["spec"]["test_code"], regression_command=proposal["regression"],
                test_driver=list(proposal["spec"].get("test_driver", ())))
    return plan


def baseline_conditions(loop, spec, regression):
    from agent.storage.recovery import environment_key
    return digest({"environment": environment_key(loop.environment), "timeout": loop.environment.timeout,
                   "original_source": loop.baseline.digest, "workspace_source": loop.evidence.current,
                   "spec": asdict(spec), "regression": regression})


def append_result(loop, response, result):
    call = response["data"]["tool_calls"][0]
    loop.state["history"].extend([response, message_to_dict(ToolMessage(
        content=json.dumps(result, ensure_ascii=False), name=call["name"], tool_call_id=call["id"]))])
    loop.state.update(response_pending=None, plan_error="", repair_count=0)


def behavior(code):
    """An AST boundary, not proof of equivalent fixtures or correct expectations."""
    tree = ast.parse(code)
    return [ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]


def spec_for(loop, code, driver=()):
    request = loop.state["request"] + ''.join('\n' + row['text'] for row in loop.state.get('user_updates', []))
    text = request + "\n" + (loop.state["acceptance"] or "")
    consumer = required_consumer(text)
    return VerificationSpec(tuple([request] +
        ([loop.state["acceptance"]] if loop.state["acceptance"] else [])), test_code=code,
        required_consumer=consumer, required_calls=requested_calls(text) if consumer else (), test_driver=driver)


def run_report(loop, snapshot, spec, directory, phase, repository_baseline=None):
    # Local execution has no OS sandbox: use a disposable copy, never the retained original.
    if isinstance(loop.environment, LocalEnvironment):
        snapshot = SourceSnapshot.capture(snapshot.root, directory / "source")
    timeout = min(loop.environment.timeout, loop.budget.remaining_time() or loop.environment.timeout)
    with snapshot.environment_like(loop.environment, directory / "runtime", read_only=True) as backend:
        report = AcceptanceRun(snapshot.root, spec, artifacts_root=directory / "runtime",
                               environment=backend, timeout=timeout,
                               repository_baseline=repository_baseline).run(phase)
        integrity = snapshot.verify_integrity()
    report["source_integrity"] = integrity
    if not integrity["valid"]:
        report.update(status="error", tests_ready=False, integrity_ok=False,
                      error_kind="integrity_error", reason="Acceptance changed its isolated source copy")
    return report


def validate(loop):
    def proposal(ref):
        data = checked(loop.root, ref)
        if data.get("candidate"):
            checked(loop.root, data["candidate"])
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
        if (release["approval_ref"] != loop.state.get("active_acceptance")
                or release["candidate_ref"] != approval.get("candidate")
                or release["plan"] != released_plan(candidate, approval)
                or release["origin_response"] != candidate["response"]
                or release["response"] != approval["review_response"]):
            raise ModelRequestNotSent("state_conflict", "Released candidate differs from its reviewed original")


def trial(loop, code, regression, response, *, driver=(), candidate=None, parent=None, reason="",
          reuse=None, superseded=None):
    spec = spec_for(loop, code, driver)
    proposal = {"spec": asdict(spec), "regression": regression, "parent": parent,
                "reason": reason, "original_source": loop.baseline.digest,
                "workspace_source": loop.evidence.current, "candidate": candidate,
                "baseline_conditions": baseline_conditions(loop, spec, regression)}
    identifier = hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()
    folder = loop.root / "acceptance" / identifier
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "proposal.json"
    if path.exists():
        raise ValueError("Identical acceptance proposal already trialled; use its trial_id")
    save(path, proposal)
    ref = {**reference(loop.root, path), "id": identifier}
    loop.state.setdefault("acceptance_trials", []).append(ref)
    loop.state.update(acceptance_trial={"ref": ref, "candidate": candidate},
                      acceptance_running=True, response_pending=None)
    loop.persist("acceptance_trial_started", {"trial_id": identifier})
    report = checked(loop.root, reuse) if reuse else run_report(
        loop, loop.baseline, spec, folder / "baseline", "baseline")
    if not loop.baseline.verify_integrity()["valid"]:
        raise ModelRequestNotSent("state_conflict", "Retained original source changed during acceptance trial")
    if reuse:
        result_ref = reuse
    else:
        result_path = folder / "baseline-result.json"
        save(result_path, report)
        result_ref = reference(loop.root, result_path)
    completion = folder / "trial.json"
    save(completion, {"proposal": ref, "result": result_ref})
    loop.state["acceptance_trials"][-1] = reference(loop.root, completion)
    loop.state["acceptance_trial"].update(baseline_result=result_ref)
    loop.state["acceptance_running"] = False
    append_result(loop, response, {"outcome": "acceptance_preflight", "trial_id": identifier,
        "source": "original, before task edits", "workspace_modified": False,
        "baseline": report, "proposal_record": "$record/" + ref["record"],
        "result_record": "$record/" + result_ref["record"],
        "selected_candidate": candidate_summary(loop, candidate),
        "superseded_candidate": candidate_summary(loop, superseded), "baseline_reused": bool(reuse),
        "next": "Inspect the real evidence, then freeze_acceptance or revise_acceptance. "
                "A runnable assertion failure may reproduce the defect; it is not proof that the expectation is correct. "
                "No proposed modification has run. freeze_acceptance releases only the selected_candidate above. "
                "revise_acceptance changes tests only and keeps that candidate; to replace its edits before the first "
                "freeze, submit a complete candidate with submit_work and review the new trial_id."})
    loop.persist("acceptance_trial_finished", {"trial_id": identifier, "status": report["status"]})


def stage_candidate(loop, plan, response):
    if loop.state.get("verification_spec") or loop.state.get("verify_command"):
        raise ValueError("Pending replacement cannot change frozen acceptance")
    validate(loop)
    current = loop.state.get("acceptance_trial")
    reuse = None
    if current:
        if loop.state.get("acceptance_running") or not current.get("baseline_result"):
            raise ModelRequestNotSent("execution_unknown", "Pending acceptance trial has no completed result")
        proposal = checked(loop.root, current["ref"])
        if proposal["workspace_source"] != loop.evidence.current:
            raise ModelRequestNotSent("state_conflict", "Workspace changed while acceptance was under review")
        report = checked(loop.root, current["baseline_result"])
        conditions = baseline_conditions(loop, spec_for(loop, plan.test_code, plan.test_driver), plan.regression_command)
        remaining = loop.budget.remaining_time()
        if (proposal["baseline_conditions"] == conditions and tests_ready_for_candidate(report)
                and (remaining is None or remaining >= loop.environment.timeout)):
            reuse = current["baseline_result"]
    trial(loop, plan.test_code, plan.regression_command, response,
          driver=plan.test_driver, candidate=candidate_record(loop, plan, response), reuse=reuse,
          parent=current["ref"]["id"] if current else None,
          superseded=current.get("candidate") if current else None,
          reason="Explicit pending candidate replacement" if current else "")


def review(loop, decision, response):
    validate(loop)
    current_trial = loop.state.get("acceptance_trial")
    current_ref = current_trial["ref"] if current_trial else loop.state.get("active_acceptance")
    if not current_ref or decision.trial_id != current_ref["id"]:
        raise ValueError("Review requires the current acceptance trial_id")
    proposal = checked(loop.root, current_ref)
    if proposal["workspace_source"] != loop.evidence.current and current_trial:
        raise ModelRequestNotSent("state_conflict", "Workspace changed while acceptance was under review")
    if decision.decision == "revise":
        if loop.state.get("verify_command"):
            raise ValueError("User fixed acceptance cannot be revised")
        if not decision.test_code.strip():
            raise ValueError("Revision requires complete test_code; no code is guessed")
        if loop.state.get("verification_spec"):
            if tests_ready_for_candidate(loop.state.get("verification", {})):
                raise ValueError("Runnable behavior failed; repair source, not frozen acceptance")
            old_code = loop.state["verification_spec"]["test_code"]
            if behavior(old_code) != behavior(decision.test_code):
                raise ValueError("Frozen revision may change module preparation only; test classes, "
                                 "helpers, fixtures, inputs and assertions must remain unchanged")
        trial(loop, decision.test_code, proposal["regression"], response,
              driver=decision.test_driver or proposal["spec"].get("test_driver", ()),
              candidate=current_trial.get("candidate") if current_trial else None,
              parent=current_ref["id"], reason=decision.reason)
        return
    if not current_trial or not current_trial.get("baseline_result"):
        raise ValueError("No completed acceptance trial can be frozen")
    if decision.test_code or decision.test_driver:
        raise ValueError("freeze_acceptance confirms existing code; use revise_acceptance to submit code")
    report = checked(loop.root, current_trial["baseline_result"])
    if not tests_ready_for_candidate(report):
        raise ValueError("Acceptance is not runnable; inspect its baseline errors and revise it before freezing")
    version = {**proposal, "baseline_result": current_trial["baseline_result"],
               "review_reason": decision.reason, "review_response": response}
    version_path = loop.root / "acceptance" / current_ref["id"] / "frozen.json"
    save(version_path, version)
    ref = {**reference(loop.root, version_path), "id": current_ref["id"]}
    loop.state.setdefault("acceptance_versions", []).append(ref)
    loop.state.update(active_acceptance=ref, verification_spec=proposal["spec"],
                      regression_command=proposal["regression"], acceptance_trial=None)
    save(loop.root / "frozen-acceptance.json", {"spec": proposal["spec"], "regression": proposal["regression"]})
    candidate_ref = current_trial.get("candidate")
    if candidate_ref:
        candidate = checked(loop.root, candidate_ref)
        loop.state["acceptance_release"] = {"plan": released_plan(candidate, proposal), "response": response,
                                            "origin_response": candidate["response"],
                                            "candidate_ref": candidate_ref, "approval_ref": ref}
    else:
        append_result(loop, response, {"outcome": "acceptance_frozen", "trial_id": ref["id"],
            "next": "Submit source repair using this acceptance; previous versions remain recorded."})
    loop.persist("acceptance_frozen", {"trial_id": ref["id"]})


def compare(loop, snapshot, active_report, folder):
    validate(loop)
    versions = loop.state.get("acceptance_versions", [])
    rows, conflicts = [], []
    for ref in versions:
        version = checked(loop.root, ref)
        baseline = checked(loop.root, version["baseline_result"])
        report = active_report if ref == loop.state.get("active_acceptance") else run_report(
            loop, snapshot, VerificationSpec(**version["spec"]), folder / ref["id"], "candidate", baseline)
        if ref != loop.state.get("active_acceptance") and (
                report.get("status") != "passed" and (tests_ready_for_candidate(baseline)
                                                      or tests_ready_for_candidate(report))):
            conflicts.append(ref["id"])
        rows.append({"trial_id": ref["id"], "baseline": baseline, "candidate": report})
    result = {"versions": rows, "conflicts": conflicts,
              "meaning": "Execution comparisons, not a proof of semantic coverage or correct expectations."}
    save(folder / "comparison.json", result)
    active_report["acceptance_comparison"] = {"record": "$record/" + (folder / "comparison.json").relative_to(loop.root).as_posix(),
                                             "conflicts": conflicts}
    if conflicts:
        active_report.update(status="failed", reason="Retained acceptance has unresolved runnable failures")
    return active_report
