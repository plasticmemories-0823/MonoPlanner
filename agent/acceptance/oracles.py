"""保守检查生成的测试，不据此证明测试覆盖完整。"""

import ast


_NORMALIZERS = {
    "str",
    "repr",
    "int",
    "float",
    "len",
    "list",
    "tuple",
    "set",
    "dict",
    "sorted",
    "round",
    "bool",
}


def oracle_issues(tree):
    helpers = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    issues = []
    for method in ast.walk(tree):
        if not isinstance(method, ast.FunctionDef) or not method.name.startswith("test_"):
            continue
        bindings = {}

        def origins(node, seen=frozenset()):
            if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
                return origins(bindings[node.id], seen | {node.id})
            found = set()
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name not in _NORMALIZERS and name not in helpers:
                    # 追踪断言值的生成器，不追踪用于构造输入的公共工厂，
                    # 例如 Symbol。
                    return {name}
            for child in ast.iter_child_nodes(node):
                found.update(origins(child, seen))
            return found

        def visit(node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bindings[target.id] = node.value
            pairs = []
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"assertEqual", "assertAlmostEqual"} and len(node.args) >= 2:
                    pairs.append((node.args[0], node.args[1]))
            if (
                isinstance(node, ast.Compare)
                and len(node.comparators) == 1
                and isinstance(node.ops[0], ast.Eq)
            ):
                pairs.append((node.left, node.comparators[0]))
            for left, right in pairs:
                shared = origins(left) & origins(right)
                if shared:
                    issues.append(
                        f"{method.name}:{node.lineno} 的实际值与预期值共同调用 {', '.join(sorted(shared))}；"
                        "不能仅用同一实现产生真值，请改为独立预期或独立执行验证。"
                    )
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(method)
    return list(dict.fromkeys(issues))
