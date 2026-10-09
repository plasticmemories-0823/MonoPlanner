"""Validate recorded test evidence and compare repository conditions."""
import json
from pathlib import Path

def _repository_result(self, report, phase):
    generated = set(report["generated_test_ids"])
    groups = {}
    for name, is_generated in (("generated", True), ("repository", False)):
        rows = [r for r in report["test_results"] if (r["id"] in generated) == is_generated]
        groups[name] = {"test_ids": [r["id"] for r in rows],
                        "counts": {status: sum(r["status"] == status for r in rows)
                                   for status in {r["status"] for r in rows}}}
    skips = {r["id"]: r["detail"] for r in report["test_results"]
             if r["status"] == "skipped" and r["id"] not in generated}
    report.update(test_groups=groups, repository_skips=skips)
    if report.get("execution_exception") or report.get("native_exit_code") not in (0, 1) or (
            report.get("native_exit_code") == 1 and not report["failures"] and not report["errors"]):
        report.update(status="error", error_kind="driver_error",
                      reason="仓库驱动非零退出与已记录测试结果不一致。")
    if phase == "baseline":
        return
    baseline = self.repository_baseline
    issue = None
    if baseline is not None:
        if (not baseline.get("test_evidence_valid") or not baseline.get("integrity_ok")
                or set(baseline.get("test_ids") or []) != set(report["test_ids"])
                or set(baseline.get("generated_test_ids") or []) != generated):
            issue = "候选测试清单或自写／仓库分类与冻结原始试跑不一致。"
        elif any(baseline.get("repository_skips", {}).get(key) != detail for key, detail in skips.items()):
            issue = "候选新增了仓库跳过项或改变了跳过原因；不能用跳过替代回归。"
    elif skips:
        issue = "仓库存在跳过项，需要冻结的原始试跑记录对照；未认定通过。"
    if issue:
        report.update(status="error", error_kind="repository_comparison", reason=issue)
    elif baseline is not None:
        report["repository_baseline_matched"] = True
        if skips and report["status"] == "passed":
            report["reason"] = "自写验收及已执行仓库测试通过；仓库跳过项与原始试跑一致，另行保留，未计为通过。"


def _read_test_result(path: Path, report: dict):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(result, dict) or not isinstance(result.get("status"), str)
                or result["status"] not in {"passed", "failed", "error"}):
            raise ValueError("无效的测试状态")
        native = "generated_test_ids" in result
        if result["status"] == "passed" and (
            report["exit_code"] != 0 or not isinstance(result.get("tests_run"), int)
            or result["tests_run"] <= 0 or any(result.get(key) != 0 for key in
                (("failures", "errors", "expected_failures", "unexpected_successes") if native else
                 ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")))
        ):
            raise ValueError("通过状态与测试证据不一致")
        ids, rows = result.get("test_ids"), result.get("test_results")
        statuses = {"passed", "failed", "error", "skipped", "expected_failure",
                    "unexpected_success", "not_run"}
        if (not isinstance(ids, list) or any(not isinstance(key, str) or not key for key in ids)
                or len(set(ids)) != len(ids) or not isinstance(rows, list)):
            raise ValueError("缺少有效的逐项测试证据")
        seen = set()
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                    or not row["id"] or row["id"] in seen
                    or not isinstance(row.get("status"), str) or row["status"] not in statuses
                    or not isinstance(row.get("detail"), str)):
                raise ValueError("逐项测试结果包含无效或重复的 ID/状态")
            seen.add(row["id"])
            exceptions = row.get("exceptions", [])
            if not isinstance(exceptions, list) or any(
                not isinstance(e, dict) or e.get("phase") not in {"test", "setup", "teardown", "unknown"}
                or e.get("origin") not in {"source", "test", "external", "unknown"}
                or type(e.get("ordinary_exception")) is not bool
                or not isinstance(e.get("type"), str) or not isinstance(e.get("message"), str)
                or not isinstance(e.get("frames"), list) or any(
                    not isinstance(f, dict) or not isinstance(f.get("file"), str)
                    or type(f.get("line")) is not int or not isinstance(f.get("function"), str)
                    or f.get("scope") not in {"source", "test", "external"}
                    for f in e["frames"]) for e in exceptions):
                raise ValueError("异常来源证据格式无效")
        if not set(ids).issubset(seen):
            raise ValueError("逐项测试结果遗漏了已收集的测试")
        if "generated_test_ids" in result:
            generated = result["generated_test_ids"]
            if (not isinstance(generated, list) or not generated
                    or len(set(generated)) != len(generated)
                    or not set(generated).issubset(set(ids))
                    or any(row["status"] == "not_run" for row in rows if row["id"] in generated)):
                raise ValueError("仓库驱动没有实际执行自写验收用例")
            report.update(generated_test_ids=generated, native_exit_code=result.get("native_exit_code"))
            for status, key in (("failed", "failures"), ("error", "errors"), ("skipped", "skipped"),
                                ("expected_failure", "expected_failures"), ("unexpected_success", "unexpected_successes")):
                if result.get(key) != sum(r["status"] == status for r in rows):
                    raise ValueError("仓库测试计数与逐项结果不一致")
        if result["status"] == "passed" and (
            set(ids) != seen or len(ids) != result["tests_run"]
            or any(row["status"] != "passed" and not (
                native and row["status"] == "skipped" and row["id"] not in generated) for row in rows)
        ):
            raise ValueError("通过状态与逐项测试证据不一致")
        for key in ("status", "reason", "tests_run", "failures", "errors", "skipped",
                    "expected_failures", "unexpected_successes"):
            if key in result:
                report[key] = result[key]
        report.update(test_ids=sorted(ids), test_results=rows, test_evidence_valid=True)
        if "execution_exception" in result:
            report["execution_exception"] = result["execution_exception"]
    except (OSError, ValueError) as exc:
        report.update(status="error", reason=f"没有有效的验收测试结果：{exc}")
