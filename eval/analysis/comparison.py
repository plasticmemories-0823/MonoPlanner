"""Compare two retained same-task runs; no model calls, no changes to source data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def summarize(directory, official):
    directory = Path(directory)
    outcome = read(directory / "result.json")
    state = read(directory / "agent-result.json")
    total, owners = Counter(), defaultdict(Counter)
    tools, previous, schemas, systems = defaultdict(Counter), {}, defaultdict(set), defaultdict(set)
    timeline, prefixes, first = [], [], {}
    for path in sorted((directory / "model-calls").glob("*.request.json")):
        request = read(path)
        response = read(path.with_name(path.name.replace(".request.", ".response.")))
        names = {t["function"]["name"] for t in request.get("tools", [])}
        owner = "preparation" if "write_test" in names else "worker" if "run_command" in names else "closing"
        data = response.get("response", {}).get("data", {})
        usage = data.get("usage_metadata") or {}
        row = Counter(calls=1)
        row["unknown_usage_calls"] = int(not all(type(usage.get(k)) is int for k in ("input_tokens", "output_tokens", "total_tokens")))
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            row[key] = usage.get(key) or 0
        cache = (usage.get("input_token_details") or {}).get("cache_read")
        row["unknown_cache_calls"] = int(type(cache) is not int)
        row["cache_read"] = cache if type(cache) is int else 0
        row["api_errors"] = int(response.get("status") != "ok")
        row["compactions"] = int(request.get("purpose") == "compaction")
        row["truncations"] = int((data.get("response_metadata") or {}).get("finish_reason") == "length")
        total.update(row)
        owners[owner].update(row)
        tool_calls = data.get("tool_calls", [])
        tools[owner].update(t["name"] for t in tool_calls)
        wire = request["wire_input"]
        schemas[owner].add(hashlib.sha256(json.dumps(wire.get("tools", []), sort_keys=True).encode()).hexdigest())
        systems[owner].add(hashlib.sha256(json.dumps([m for m in wire["messages"] if m["role"] == "system"], sort_keys=True).encode()).hexdigest())
        if owner not in first:
            first[owner] = {**row, "call": path.name}
        if request.get("purpose") in {"work", "delivery"}:
            if owner in previous:
                old = previous[owner]
                prefixes.append({"call": path.name, "role": owner,
                    "preserved": old["messages"] == wire["messages"][:len(old["messages"])] and old.get("tools") == wire.get("tools")})
            previous[owner] = wire
        timeline.append({"call": request.get("call_id"), "role": owner, "purpose": request.get("purpose"),
            "started_utc": request["started_utc"], "seconds": response.get("elapsed_seconds"), **row,
            "tools": [{"name": t["name"], "args": t["args"]} for t in tool_calls]})
    assert total["calls"] == outcome["sent_model_calls"]
    assert total["total_tokens"] == outcome["reported_tokens"]
    for group in (total, *owners.values()):
        group["uncached_input_tokens"] = group["input_tokens"] - group["cache_read"]
        group["cache_hit_percent"] = (100 * group["cache_read"] / group["input_tokens"]
            if group["input_tokens"] and not group["unknown_cache_calls"] else None)
    grading = [read(p)[outcome["instance_id"]] for p in Path(official).rglob("report.json") if outcome["instance_id"] in read(p)]
    return {"task": outcome["instance_id"], "directory": str(directory), "totals": dict(total),
        "roles": {k: dict(v) for k,v in owners.items()}, "tools": {k: dict(v) for k,v in tools.items()},
        "task_seconds": outcome["elapsed_seconds"], "agent_status": outcome["agent_status"],
        "internal_verification": outcome.get("internal_verification"), "official": grading,
        "patch_bytes": outcome["patch_bytes"], "internal_tests": len((state.get("verification") or {}).get("test_results") or []),
        "prefix_pairs": prefixes, "preserved_prefix_pairs": sum(p["preserved"] for p in prefixes),
        "schema_versions": {k:len(v) for k,v in schemas.items()}, "system_versions": {k:len(v) for k,v in systems.items()},
        "first_role_calls": first, "timeline": timeline}


def main():
    parser = argparse.ArgumentParser()
    for key in ("old", "new", "old-official", "new-official", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    old, new = summarize(args.old, args.old_official), summarize(args.new, args.new_official)
    assert read(Path(args.old)/"task.json") == read(Path(args.new)/"task.json"), "Task inputs differ"
    fields = ("model", "max_steps", "task_token_limit", "token_limit", "model_policy", "task_timeout", "work_steps", "max_rounds")
    old_cfg, new_cfg = read(Path(args.old).parent/"config.json"), read(Path(args.new).parent/"config.json")
    controls = {k: old_cfg[k] == new_cfg[k] for k in fields}
    assert all(controls.values()), controls
    output = {"U9": old, "U11": new, "same_controls": controls,
              "limitations": ["One attempt per version, combined engineering changes; not isolated per-feature gains.",
                              "Provider-side cache is best effort; compare total and uncached tokens as well as ratio."]}
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, item in (("U9",old),("U11",new)):
        print(label, json.dumps({**item["totals"], "seconds":item["task_seconds"], "status":item["agent_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
