"""汇总保存的证据，不能根据 Agent 的文字回复推断官方通过。"""

import argparse
from collections import Counter
import json
from pathlib import Path

from eval.recording import write_json


def summarize(run_dir, official_reports=()):
    run_dir = Path(run_dir).resolve()
    summaries = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    official, test_completed = {}, set()
    for path in official_reports:
        report = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if "resolved_ids" not in report:
            raise ValueError(f"Not an official aggregate evaluation report: {path}")
        test_completed.update(report.get("completed_ids", []))
        for key, status in (
            ("resolved_ids", "resolved"),
            ("unresolved_ids", "unresolved"),
            ("error_ids", "grading_error"),
            ("empty_patch_ids", "empty_patch"),
        ):
            for instance_id in report.get(key, []):
                previous = official.get(instance_id)
                if previous and previous != status:
                    # 部分评测框架版本会把空补丁也列为未解决。
                    if {previous, status} <= {"unresolved", "empty_patch"}:
                        status = "empty_patch"
                    else:
                        raise ValueError(f"Conflicting official results for {instance_id}")
                official[instance_id] = status
    rows, tools = [], Counter()
    for saved in summaries:
        row = dict(saved)
        row["official_status"] = official.get(row["instance_id"], "not_scored")
        row["official_resolved"] = (
            True
            if row["official_status"] == "resolved"
            else False if row["official_status"] in {"unresolved", "empty_patch"} else None
        )
        input_sizes, cache_tokens = [], 0
        phase_calls, phase_tokens = Counter(), Counter()
        for path in sorted((run_dir / row["instance_id"] / "model-calls").glob("*.response.json")):
            call = json.loads(path.read_text(encoding="utf-8"))
            response = call.get("response", {}).get("data", {})
            usage = response.get("usage_metadata") or {}
            request_path = path.with_name(path.name.replace(".response.json", ".request.json"))
            if request_path.exists():
                request = json.loads(request_path.read_text(encoding="utf-8"))
                system = request["messages"][0]["data"]["content"]
                phase = (
                    "report"
                    if "【当前阶段：最终汇报】" in system
                    else (
                        "preparation"
                        if "【当前阶段：任务识别和验收准备】" in system
                        else "review" if "【当前阶段：验收复查】" in system else "worker"
                    )
                )
                phase_calls[phase] += 1
                phase_tokens[phase] += usage.get("total_tokens", 0) or 0
            if isinstance(usage.get("input_tokens"), int):
                input_sizes.append(usage["input_tokens"])
            cache_tokens += (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
            tools.update(tool["name"] for tool in response.get("tool_calls", []))
        row["max_reported_input_tokens"] = max(input_sizes, default=None)
        row["reported_cache_read_tokens"] = cache_tokens
        row["phase_model_calls"] = dict(phase_calls)
        row["phase_reported_tokens"] = dict(phase_tokens)
        row["official_test_completed"] = row["instance_id"] in test_completed
        rows.append(row)
    scored = [row for row in rows if row["official_resolved"] is not None]
    resolved = sum(row["official_resolved"] is True for row in rows)
    return {
        "run_directory": str(run_dir),
        "official_reports": [str(Path(p).resolve()) for p in official_reports],
        "selected_tasks": len(rows),
        "officially_scored_tasks": len(scored),
        "official_test_completed_tasks": sum(row["official_test_completed"] for row in rows),
        "officially_resolved_tasks": resolved,
        # 缺少评分结果不能被解释为修复失败，也不能被解释为成功。
        "complete_sample_resolve_rate": (
            resolved / len(rows) if len(scored) == len(rows) and rows else None
        ),
        "internal_accepts": sum(row["agent_status"] == "verified_success" for row in rows),
        "false_accepts_against_official_tests": sum(
            row["agent_status"] == "verified_success" and row["official_resolved"] is False
            for row in rows
        ),
        "reported_tokens": sum(row.get("reported_tokens", 0) for row in rows),
        "sent_model_calls": sum(row.get("sent_model_calls", 0) for row in rows),
        "unknown_usage_calls": sum(row.get("unknown_usage_calls", 0) for row in rows),
        "tool_requests": dict(tools),
        "tasks": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--official-report", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Use a new output file; previous analyses are retained.")
    analysis = summarize(args.run_dir, args.official_report)
    write_json(output, analysis)
    print(
        json.dumps(
            {key: value for key, value in analysis.items() if key != "tasks"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
