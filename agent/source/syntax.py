"""静态 Python 符号、导入与调用解析；不执行源码。"""

import ast
from pathlib import PurePosixPath


def _module(path):
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts.pop(0)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return parent + "." + node.attr if parent else ""
    return ""


def _parse(data, path):
    tree = ast.parse(data, filename=path)
    module = _module(path)
    symbols, imports, calls = [], [], []

    def definitions(body, prefix=""):
        for node in body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = prefix + node.name
            if isinstance(node, ast.ClassDef):
                signature = (
                    "class " + node.name + "(" + ", ".join(ast.unparse(b) for b in node.bases) + ")"
                )
            else:
                signature = (
                    ("async " if isinstance(node, ast.AsyncFunctionDef) else "")
                    + "def "
                    + node.name
                )
                signature += "(" + ast.unparse(node.args) + ")"
                if node.returns:
                    signature += " -> " + ast.unparse(node.returns)
            doc = (ast.get_docstring(node) or "").splitlines()
            symbols.append(
                {
                    "name": name,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "signature": signature[:240],
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                    "doc": doc[0][:160] if doc else "",
                    "decorators": [ast.unparse(d)[:80] for d in node.decorator_list[:4]],
                }
            )
            if len(symbols) < 160:
                definitions(node.body, name + ".")
            if len(symbols) >= 160:
                break

    definitions(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "target": alias.name,
                        "binding": alias.asname or alias.name.split(".")[0],
                        "bound_target": alias.name if alias.asname else alias.name.split(".")[0],
                        "line": node.lineno,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            package = (
                module.split(".")
                if PurePosixPath(path).name == "__init__.py"
                else module.split(".")[:-1]
            )
            prefix = package[: max(0, len(package) - node.level + 1)] if node.level else []
            base = ".".join([*prefix, *([node.module] if node.module else [])])
            for alias in node.names:
                target = ".".join(filter(None, (base, alias.name)))
                imports.append(
                    {
                        "target": target,
                        "binding": alias.asname or alias.name,
                        "bound_target": target,
                        "line": node.lineno,
                    }
                )
    aliases = {}
    for item in imports:
        aliases.setdefault(item["binding"], set()).add(item["bound_target"])
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if not name:
            continue
        head, _, tail = name.partition(".")
        targets = [
            base + ("." + tail if tail else "") for base in aliases.get(head, {module + "." + head})
        ]
        calls.append({"name": name, "targets": sorted(targets), "line": node.lineno})
        if len(calls) >= 300:
            break
    doc = (ast.get_docstring(tree) or "").splitlines()
    return {
        "module": module,
        "doc": doc[0][:160] if doc else "",
        "symbols": symbols,
        "imports": imports[:160],
        "calls": calls,
        "error": None,
    }
