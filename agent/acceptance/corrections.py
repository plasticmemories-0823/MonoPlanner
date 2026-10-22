"""按声明原样替换自拟验收材料；证据和需求边界由调用方独立核验。"""

from copy import deepcopy
import json


FIELDS = ("test_code", "test_driver", "execution_checks", "regression_command")


def materials(spec, regression):
    return {
        "test_code": spec["test_code"],
        "test_driver": deepcopy(list(spec.get("test_driver", ()))),
        "execution_checks": deepcopy(list(spec.get("execution_checks", ()))),
        "regression_command": regression or "",
    }


def protected(spec):
    """自拟材料之外的原始需求、固定命令和已确认约束不可被此协议改写。"""
    return json.loads(json.dumps({key: value for key, value in spec.items()
                                  if key not in FIELDS}))


def rewrite(text, items):
    spans = []
    for item in items:
        start = text.find(item.before)
        if (not item.before and text) or start < 0 or (
            item.before and text.find(item.before, start + 1) >= 0
        ):
            raise ValueError("Each before snippet must have one exact unique match in the active frozen material")
        if item.before == item.after:
            raise ValueError("Each correction must change its declared material")
        spans.append((start, start + len(item.before), item.after))
    spans.sort()
    if any(left[1] > right[0] or left[0] == right[0]
           for left, right in zip(spans, spans[1:])):
        raise ValueError("Acceptance correction snippets overlap")
    result = text
    for start, end, replacement in reversed(spans):
        result = result[:start] + replacement + result[end:]
    return result


def apply(before, items):
    """文本使用精确片段；列表使用完整 JSON 前后值，不猜测未声明的改动。"""
    result = deepcopy(before)
    grouped = {}
    for item in items:
        if item.field not in FIELDS:
            raise ValueError("Corrections may target model-generated acceptance fields only")
        grouped.setdefault(item.field, []).append(item)
    for field, changes in grouped.items():
        if isinstance(before[field], str):
            result[field] = rewrite(before[field], changes)
            continue
        if len(changes) != 1:
            raise ValueError("Use one complete before/after JSON array correction for " + field)
        item = changes[0]
        try:
            old, new = json.loads(item.before), json.loads(item.after)
        except (ValueError, TypeError) as error:
            raise ValueError(field + " corrections require complete JSON arrays in before and after") from error
        if not isinstance(old, list) or not isinstance(new, list) or old != before[field]:
            raise ValueError(field + " before must equal its complete active frozen JSON array")
        if old == new:
            raise ValueError("Each correction must change its declared material")
        result[field] = new
    return result


def project(current, base, items):
    """历史投射须绑定完整基准字段，不能因同一短片段出现就改到无关旧检查。"""
    for field in {item.field for item in items}:
        old, expected = current[field], base[field]
        if old == expected:
            continue
        if (isinstance(old, str) and isinstance(expected, str) and expected
                and old.count(expected) == 1):
            continue  # 历史版本额外的准备条件原样保留。
        raise ValueError("Historical acceptance material does not match the correction base: " + field)
    return apply(current, items)
