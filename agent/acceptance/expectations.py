"""保守识别预期字面量；不能识别的 Python 写法不开放修订。"""

import ast


EQUALITY_ASSERTIONS = frozenset({
    "assertEqual", "assertListEqual", "assertTupleEqual", "assertDictEqual",
    "assertSetEqual", "assertCountEqual", "assertMultiLineEqual",
    "assertSequenceEqual", "assertAlmostEqual",
})
BOUNDARY_ERROR = (
    "Frozen correction may change expected literals only; keep inputs, tested calls, "
    "assertion types, test cases and all other code exactly unchanged"
)


def _literal(node):
    if isinstance(node, ast.Constant):
        return type(node.value) in {str, bytes, int, float, complex, bool, type(None)}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(key is not None and _literal(key) and _literal(value)
                   for key, value in zip(node.keys, node.values))
    return (
        isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in {int, float, complex}
    )


def _walk(node, path=()):
    yield path, node
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield from _walk(value, (*path, field))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, ast.AST):
                    yield from _walk(child, (*path, field, index))


def _scope_nodes(function):
    """嵌套作用域中的断言不能授权外层变量修订。"""
    def visit(node):
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp,
                                 ast.GeneratorExp)):
            for child in ast.iter_child_nodes(node):
                yield from visit(child)

    for statement in function.body:
        yield from visit(statement)


def _assertion_pair(node, receiver):
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
        if (isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == receiver
                and call.func.attr in EQUALITY_ASSERTIONS and len(call.args) >= 2
                and not any(isinstance(arg, ast.Starred) for arg in call.args)):
            return call.args[:2]
    if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)):
        return [node.test.left, node.test.comparators[0]]
    return None


def _literal_assignment(statement):
    return (isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name) and _literal(statement.value))


def _linear_blocks(statements):
    """每个 with 主体单独分析，不猜测异常抑制后的跨块绑定或分支合流。"""
    yield statements
    for statement in statements:
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            yield from _linear_blocks(statement.body)


def _local_expected_literals(function, receiver):
    parameters = {node.arg for node in ast.walk(function.args) if isinstance(node, ast.arg)}
    assignments, permitted, used = {}, {}, set()
    for statements in _linear_blocks(function.body):
        current = {}
        for statement in statements:
            if _literal_assignment(statement) and statement.targets[0].id not in parameters:
                target = statement.targets[0]
                current[target.id] = statement
                assignments.setdefault(target.id, []).append(statement)
                permitted.setdefault(target.id, set()).add(target)
                continue

            def possible(node):
                return _literal(node) or isinstance(node, ast.Name) and node.id in current

            pair = _assertion_pair(statement, receiver)
            expected = None
            if pair and possible(pair[0]) != possible(pair[1]):
                expected = pair[0] if possible(pair[0]) else pair[1]
                if isinstance(expected, ast.Name):
                    used.add(current[expected.id])
                    permitted[expected.id].add(expected)
            # 输入、别名、修改、嵌套控制流或其它用途不能成为后续断言的绑定依据。
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and node is not expected:
                    current.pop(node.id, None)

    nodes = list(ast.walk(function))
    dynamic = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id in {"locals", "globals", "vars", "eval", "exec"}
                  for node in nodes)
    result = []
    for name, bindings in assignments.items():
        refs = {node for node in nodes if isinstance(node, ast.Name) and node.id == name}
        other_binding = any(
            isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names
            or isinstance(node, ast.ExceptHandler) and node.name == name
            or isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name
            or isinstance(node, ast.MatchMapping) and node.rest == name
            or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node is not function and node.name == name
            or isinstance(node, ast.arg) and node.arg == name
            or isinstance(node, ast.alias) and (node.asname or node.name.split('.')[0]) == name
            for node in nodes
        )
        if not dynamic and not other_binding and refs == permitted[name]:
            result.extend(binding.value for binding in bindings if binding in used)
    return result


def _expected_literals(function, method=False):
    nodes = list(_scope_nodes(function))
    receiver = function.args.args[0].arg if method and function.args.args else None
    parameters = {node.arg for node in ast.walk(function.args) if isinstance(node, ast.arg)}
    literal_names = {node.targets[0].id for node in nodes if _literal_assignment(node)}
    literal_names.difference_update(parameters)
    direct = _local_expected_literals(function, receiver)

    def possible(node):
        return _literal(node) or isinstance(node, ast.Name) and node.id in literal_names

    for node in nodes:
        pair = _assertion_pair(node, receiver)
        if pair is None or possible(pair[0]) == possible(pair[1]):
            continue  # 两边都是常量时，不能判定哪边是测试输入。
        expected = pair[0] if possible(pair[0]) else pair[1]
        if _literal(expected):
            direct.append(expected)
    return direct


def _targets(code):
    tree = ast.parse(code)
    paths = {id(node): path for path, node in _walk(tree)}
    modules, cases = set(), set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.asname or alias.name for alias in node.names
                           if alias.name == "unittest")
        if isinstance(node, ast.ImportFrom) and node.module == "unittest" and not node.level:
            cases.update(alias.asname or alias.name for alias in node.names
                         if alias.name == "TestCase")
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append((node, False))
        elif (isinstance(node, ast.ClassDef) and len(node.bases) == 1
              and not node.decorator_list and not node.keywords):
            base = node.bases[0]
            known = (isinstance(base, ast.Name) and base.id in cases or
                     isinstance(base, ast.Attribute) and base.attr == "TestCase"
                     and isinstance(base.value, ast.Name) and base.value.id in modules)
            overridden = any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                             and item.name in EQUALITY_ASSERTIONS for item in node.body)
            if known and not overridden:
                functions.extend((item, True) for item in node.body
                                 if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))
    targets = {}
    for function, method in functions:
        if function.name.startswith("test") and not function.decorator_list:
            for node in _expected_literals(function, method):
                targets[paths[id(node)]] = node
    return targets


def _masked(code, targets):
    """AST 列号是 UTF-8 字节偏移；原样保留其它字节、注释和换行。"""
    raw = code.encode("utf-8")
    offsets, total = [], 0
    for line in raw.splitlines(keepends=True):
        offsets.append(total)
        total += len(line)
    spans = sorted((offsets[node.lineno - 1] + node.col_offset,
                    offsets[node.end_lineno - 1] + node.end_col_offset, path)
                   for path, node in targets.items())
    parts, previous = [], 0
    for start, end, path in spans:
        parts.extend([raw[previous:start], repr(path).encode("ascii")])
        previous = end
    parts.append(raw[previous:])
    return b"".join(parts)


def validate_change(before, after):
    try:
        old, new = _targets(before), _targets(after)
        if not old or old.keys() != new.keys() or _masked(before, old) != _masked(after, new):
            raise ValueError(BOUNDARY_ERROR)
    except (SyntaxError, RecursionError, UnicodeError) as exc:
        raise ValueError(BOUNDARY_ERROR) from exc


def rewrite(code, corrections):
    spans = []
    for item in corrections:
        start = code.find(item.before)
        if not item.before or start < 0 or code.find(item.before, start + 1) >= 0:
            raise ValueError("Each old expectation snippet must have one exact unique match")
        if item.before == item.after:
            raise ValueError("Each correction must change its declared expectation")
        spans.append((start, start + len(item.before), item.after))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("Expectation correction snippets overlap")
    revised = code
    for start, end, after in reversed(spans):
        revised = revised[:start] + after + revised[end:]
    validate_change(code, revised)
    return revised
