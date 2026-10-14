"""静态 Python 符号与词法查询提示；不执行源码。"""

import ast
from pathlib import PurePosixPath
import re


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


def _tokens(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1_\2", text)
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9]*", text)
        if len(word) > 1
        and word.lower()
        not in {"the", "and", "for", "from", "with", "def", "class", "test", "tests", "fix", "src"}
    }


def _is_test(path):
    p = PurePosixPath(path)
    return "tests" in p.parts or p.name.startswith("test_") or p.name.endswith("_test.py")


def query_identifiers(query):
    """区分大小写的代码名称；字面量和偶然出现的单字母变量不视为 API。"""
    query = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", " ", query)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
    quoted = re.findall(r"(?<!`)`([^`\n]+)`(?!`)", query)
    common = {
        "has",
        "for",
        "and",
        "the",
        "this",
        "that",
        "when",
        "where",
        "with",
        "from",
        "into",
        "is",
        "are",
        "not",
        "use",
        "fix",
        "please",
        "can",
        "you",
        "in",
        "on",
        "it",
        "a",
        "an",
        "i",
        "out",
        "true",
        "false",
        "none",
        "exists",
    }
    names = {
        name
        for text in quoted
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
        if len(name) > 1 and name.lower() not in common
    }
    for word in words:
        if word.lower() in common or (len(word) == 1 and query.strip() != word):
            continue
        if "_" in word or any(c.isupper() for c in word[1:]) or word[0].isupper():
            names.add(word)
        elif len(words) <= 3 or re.search(r"\b" + re.escape(word) + r"\s*\(", query):
            names.add(word)
    return names


def query_focus(query):
    """优先选择问题中第一个明确的 API，包括嵌套调用。

    跳过仅属描述文字的外层包装；c(f(x)) 先关注 c，再关注其参数 f。
    这只是导航提示，不用于自动分类任务或验收。"""
    for line in query.splitlines():
        names = query_identifiers(line)
        if not names:
            continue
        call = re.search(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", line)
        primary = call.group(1).split(".")[-1] if call else None
        return names, primary
    return set(), None


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
