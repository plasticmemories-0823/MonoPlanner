"""开工方案与人类确认的原件交接；不接受工作模型自批的验收范围。"""

from copy import deepcopy
import json
from pathlib import Path

from agent.storage.records import digest, save
from agent.task.requirements import DeliveryTarget, confirmed, checked


def messages(store, conversation):
    store.get(conversation)
    with store.connection() as db:
        return [json.loads(row[0]) for row in db.execute(
            "SELECT data FROM app_messages WHERE conversation=? ORDER BY seq", (conversation,),
        )]


def source_messages(store, conversation, source_ids):
    if (not isinstance(source_ids, list) or len(source_ids) > 128
            or any(not isinstance(key, str) for key in source_ids)):
        raise ValueError('Requirement source IDs must be a bounded list of saved message IDs')
    rows = [row for row in messages(store, conversation)
            if row["source"] == "user" and row["type"] == "human"]
    if not source_ids:
        return {"sources": [{"id": row["id"], "preview": row["content"][:160]} for row in rows]}
    selected = {row["id"]: row for row in rows}
    if len(set(source_ids)) != len(source_ids) or any(key not in selected for key in source_ids):
        raise ValueError("Requirement references must identify distinct saved user messages")
    return {"sources": [{"id": key, "role": "user", "text": selected[key]["content"]}
                        for key in source_ids]}


def latest(store, conversation):
    work = store.work(store.get(conversation)["work_id"])
    from agent.application.store import TERMINAL

    if not work or work['status'] in TERMINAL:
        return None
    value = work.get("confirmed_requirements")
    controls = store.commands(conversation=conversation, task=work["task_id"],
                              states=("pending", "running", "applied", "deferred"))
    for row in controls:
        value = row["value"].get("confirmed_requirements") or value
    return value


def work_for_proposal(store, conversation, proposal_id):
    """确认按钮即使换了命令 ID，也不能再次启动同一份已消费方案。"""
    for work in store.work_for_conversation(conversation):
        value = work.get("confirmed_requirements") or {}
        if value.get("confirmation", {}).get("proposal_id") == proposal_id:
            return work
        for row in store.commands(conversation=conversation, task=work["task_id"],
                                  states=("pending", "running", "applied", "unknown", "deferred")):
            packet = row["value"].get("confirmed_requirements") or {}
            if packet.get("confirmation", {}).get("proposal_id") == proposal_id:
                return work
    return None


def prepare(store, conversation, command, *, project_path, create, request_message_ids, targets,
            withdrawals=()):
    from agent.application.project_location import resolve_location

    location = resolve_location(store.get(conversation)["repo"], project_path, create)
    if not request_message_ids:
        raise ValueError("Select complete original user messages before preparing work")
    source_messages(store, conversation, request_message_ids)
    parent = latest(store, conversation)
    work = store.work(store.get(conversation)['work_id'])
    selected_ids = list(request_message_ids)
    if parent and work:
        # 尚未确认的原始补充也必须保留，模型不能通过漏选消息来丢掉它们。
        for row in store.commands(conversation=conversation, task=work['task_id'],
                                  states=('pending', 'running', 'applied', 'deferred')):
            if row['kind'] == 'supplement' and not row['value'].get('confirmed_requirements'):
                if row['id'] not in selected_ids:
                    selected_ids.append(row['id'])
    sources = source_messages(store, conversation, selected_ids)["sources"]
    merged_sources = {row["id"]: row for row in (parent["sources"] if parent else [])}
    merged_sources.update({row["id"]: row for row in sources})
    merged = {row["id"]: row for row in (parent["targets"] if parent else [])}
    for index, raw in enumerate(targets):
        try:
            target = DeliveryTarget.model_validate(raw).model_dump()
        except ValueError as exc:
            raise ValueError(f"targets[{index}]: {exc}") from exc
        if parent and target["id"] in merged and target != merged[target["id"]]:
            previous_ids = {row["id"] for row in parent["sources"]}
            if not any(q["source_id"] not in previous_ids for q in target["quotes"]):
                raise ValueError("A changed target requires an exact new user-change quotation")
        merged[target["id"]] = target
    removed = []
    for row in withdrawals:
        if not isinstance(row, dict) or set(row) != {'target_id', 'source_id', 'quote'}:
            raise ValueError('A withdrawal requires a target ID and an exact new user-change quote')
        key = row['target_id']
        prior_ids = {item['id'] for item in parent['sources']} if parent else set()
        source = merged_sources.get(row['source_id'])
        if (not parent or key not in merged or key in {item['id'] for item in targets}
                or row['source_id'] in prior_ids or not source or not row['quote']
                or row['quote'] not in source['text']):
            raise ValueError('Only an explicit newly quoted user change can withdraw an existing target')
        del merged[key]
        removed.append(dict(row))
    if not merged:
        raise ValueError("Prepare at least one concrete delivery target")
    # 此时只验证引用与接口；最终确认只能由之后保存的人类消息完成。
    source_order = {row['id']: index for index, row in enumerate(messages(store, conversation))}
    sources = sorted(merged_sources.values(), key=lambda row: source_order[row['id']])
    request = "\n\n".join(row["text"] for row in sources)
    provisional = confirmed(request, list(merged.values()), sources=sources,
                            confirmation={"id": "unconfirmed", "text": "not authorized"},
                            parent=parent)
    proposal = {
        "conversation": conversation, "request": request, "sources": provisional["sources"],
        "targets": provisional["targets"], "parent": parent,
        "project_path": location["path"], "create": create,
        "prepared_after": command["id"],
        "withdrawals": removed,
    }
    identifier = digest(proposal)
    path = store.root / "work-proposals" / conversation / (identifier + ".json")
    if path.exists():
        if json.loads(path.read_text("utf-8")) != proposal:
            raise ValueError("Saved work proposal changed")
    else:
        save(path, proposal)
    lines = ["待确认的工作方案", "", "项目：" + location["path"]]
    for removal in removed:
        lines += ['', '按新需求取消交付物：' + removal['target_id'], '变更原文：' + removal['quote']]
    for target in proposal["targets"]:
        lines += ["", "交付物：" + target["id"],
                  "类型／运行环境：" + target["artifact_type"] + " / " + target["target_runtime"],
                  "入口：" + target["entry"], "需求行为：" + target["behavior"],
                  "输入范围：" + target["input_scope"]]
        if target["expression"]:
            lines += ["原始导入：" + target["imports"], "实际产物调用：" + target["expression"],
                      "输入构造：" + (target["setup"] or "无额外构造")]
        if target["required_values"]:
            lines.append("必须执行的输入：" + json.dumps(target["required_values"], ensure_ascii=False))
    lines += ["", "点击下方确认按钮后按这些要求开始；也可以先在聊天中修改方案。"]
    store.publish(conversation, "work-proposal:" + identifier, "\n".join(lines), source="work_proposal")
    return {"status": "awaiting_confirmation", "proposal_id": identifier,
            "project_path": location["path"], "create": create,
            "request_message_ids": [row["id"] for row in sources]}


def load_proposal(store, conversation, proposal_id):
    from agent.application.store import identity
    from agent.source.snapshots import _safe_file

    conversation = identity(conversation)
    if not isinstance(proposal_id, str) or len(proposal_id) != 64 or any(
        char not in "0123456789abcdef" for char in proposal_id
    ):
        raise ValueError("A saved delivery proposal must be confirmed before starting work")
    path = _safe_file(store.root, Path('work-proposals') / conversation / (proposal_id + '.json'))
    proposal = json.loads(path.read_text("utf-8"))
    if digest(proposal) != proposal_id or proposal["conversation"] != conversation:
        raise ValueError("Work proposal identity changed")
    return proposal


def accept(store, conversation, command, proposal_id, project_path, create):
    proposal = load_proposal(store, conversation, proposal_id)
    rows = messages(store, conversation)
    indices = {row["id"]: index for index, row in enumerate(rows)}
    confirmation = next((row for row in rows if row["id"] == command["id"]), None)
    if (command.get("kind") != "confirm_work" or command.get("proposal_id") != proposal_id
            or not confirmation or confirmation["source"] != "user"
            or confirmation["type"] != "human"
            or indices[command["id"]] <= indices.get("work-proposal:" + proposal_id, len(rows))):
        raise ValueError("Starting requires a later saved human confirmation of this proposal")
    if proposal["project_path"] != project_path or proposal["create"] != create:
        raise ValueError("Confirmed project location differs from the proposal")
    current = latest(store, conversation)
    old = proposal["parent"]
    work = store.work(store.get(conversation)['work_id'])
    if old and work:
        included = {row['id'] for row in proposal['sources']}
        for row in store.commands(conversation=conversation, task=work['task_id'],
                                  states=('pending', 'running', 'applied', 'deferred')):
            if (row['kind'] == 'supplement' and not row['value'].get('confirmed_requirements')
                    and row['id'] not in included):
                raise ValueError('A newer original requirement supplement needs a new proposal')
    if (current or {}).get("digest") != (old or {}).get("digest"):
        # 同一确认的派发去重由原 create_work 处理；不另造一份需求版本。
        if current and current["confirmation"]["id"] == command["id"]:
            return current, proposal["request"]
        raise ValueError("Requirement proposal is stale after a newer confirmed change")
    originals = source_messages(store, conversation, [row["id"] for row in proposal["sources"]])["sources"]
    if any(a["text"] != b["text"] for a, b in zip(originals, proposal["sources"])):
        raise ValueError("Original requirement messages changed after proposal")
    value = confirmed(proposal["request"], proposal["targets"], sources=originals,
                      confirmation={"id": command["id"], "text": confirmation["content"],
                                    "proposal_id": proposal_id}, parent=old)
    checked(value, proposal["request"])
    return deepcopy(value), proposal["request"]
