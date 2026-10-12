"""Bounded startup evidence, selected from source names without executing project code."""
import ast
from pathlib import PurePosixPath
import re

from agent.source.paging import merged
from agent.source.syntax import _tokens, query_identifiers

MAX_LINES = 720
MAX_CHARS = 24000
RUNNERS = ('runtests.py', 'tests/runtests.py', 'test/runtests.py', 'bin/test')
CONFIGS = ('pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg')


def supplied_locations(request, names):
    lookup = {}
    for name in names:
        lookup.setdefault(name.casefold(), []).append(name)
    references = []
    patterns = (
        r'(?<![\w./\\:])(?P<path>[\w-]+(?:[/\\][\w.-]+)*\.(?:py|js|ts|tsx|go|rs|java|c|cpp|h))(?::(?P<line>\d+)(?:-\d+)?)?',
        r'(?<![\w./\\:])(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+):(?P<line>\d+)(?:-\d+)?',
    )
    for pattern in patterns:
        for match in re.finditer(pattern, request):
            value = match.groupdict()
            path = value.get('path') or value['module'].replace('.', '/') + '.py'
            matches = lookup.get(path.replace('\\', '/').casefold(), [])
            if len(matches) == 1 and matches[0] not in [r['path'] for r in references]:
                references.append({'path': matches[0], 'line': int(value.get('line') or 1),
                                   'reference': match.group(0)})
    return references[:3]


def python_tree(executor, name):
    try:
        raw = executor.source_bytes(name)
        return ast.parse(raw) if len(raw) <= 512 * 1024 else None
    except (SyntaxError, UnicodeError, ValueError, RecursionError):
        return None


def implementation_ranges(tree, line):
    definitions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.lineno <= line <= n.end_lineno] if tree else []
    function = min(definitions, key=lambda n: n.end_lineno - n.lineno, default=None)
    if function is None or function.end_lineno - function.lineno >= 120:
        return [[max(0, line - 26), line + 35]]
    spans = [[function.lineno - 1, function.end_lineno]]
    # At most two direct self.method definitions in the same class. Static links
    # locate missing implementation text; they do not establish runtime behavior.
    owner = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and function in n.body), None)
    if owner:
        calls = {n.func.attr for n in ast.walk(function) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == 'self'}
        related = [n for n in owner.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name in calls and n is not function and n.end_lineno - n.lineno < 80]
        spans.extend([n.lineno - 1, n.end_lineno] for n in related[:2])
    return merged(spans)


def related_suites(request, locations, names):
    identifiers = _tokens(' '.join(query_identifiers(request)))
    calls = {m.lower() for m in re.findall(r'\b([A-Za-z_]\w*)\s*\(', request)}
    directories = {str(PurePosixPath(n).parent) for n in names
                   if PurePosixPath(n).name in {'models.py', 'tests.py', 'conftest.py'}
                   or PurePosixPath(n).name.startswith('test_')}
    scored = []
    for directory in directories:
        path = PurePosixPath(directory)
        if not {'test', 'tests', 'testing'}.intersection(path.parts):
            continue
        near = any(path == PurePosixPath(p['path']).parent / 'tests' for p in locations)
        score = 60 * near + 40 * (path.name.lower() in calls) + 12 * len(_tokens(path.name) & identifiers)
        if score:
            scored.append((score, directory))
    best = max((score for score, _ in scored), default=0)
    return [name for score, name in sorted(scored, key=lambda v: (-v[0], v[1]))
            if score >= best * .8][:2]


def referenced_settings(tree, runner, names):
    """Only literal defaults used by argument/environment setup, never guessed modules."""
    result = []
    for node in ast.walk(tree) if tree else []:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {'add_argument', 'setdefault', 'get'}:
            continue
        selected = [kw.value for kw in node.keywords if kw.arg == 'default']
        if node.func.attr in {'get', 'setdefault'} and len(node.args) > 1:
            if isinstance(node.args[0], ast.Constant) and 'SETTINGS' in str(node.args[0].value):
                selected.append(node.args[1])
        for default in selected:
            for value in ast.walk(default):
                if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                    continue
                if not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', value.value):
                    continue
                module = value.value.replace('.', '/') + '.py'
                for path in (str(PurePosixPath(runner).parent / module), module):
                    if path in names and path not in result:
                        result.append(path)
    return result[:2]


def runner_setup_ranges(tree):
    """Expose bounded, complete native loading rules rather than head/tail guesses."""
    if tree is None:
        return []
    signals = {'INSTALLED_APPS', 'DJANGO_SETTINGS_MODULE', 'configure', 'pytest_plugins'}
    spans = []
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        identifiers = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
        identifiers.update(node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute))
        if identifiers & signals and function.end_lineno - function.lineno < 160:
            spans.append([function.lineno - 1, function.end_lineno])
    return spans[:2]


def work_materials(executor, request):
    names = set(executor.inventory())
    locations = supplied_locations(request, names)
    suites = related_suites(request, locations, names)
    result = {'source_locations': locations, 'test_suite_candidates': suites,
              'selection': 'Exact supplied references and bounded lexical test-directory matches; not coverage claims.',
              'test_entrypoints': [], 'reads': []}
    lines_left, chars_left, selected = MAX_LINES, MAX_CHARS, {}

    def include(name, spans, purpose):
        nonlocal lines_left, chars_left
        if name not in names or name in selected:
            return
        selected[name] = purpose
        for start, stop in spans:
            count = min(stop - start, lines_left, 200)
            if count <= 0 or chars_left <= 0:
                break
            page = executor.read(name, start, count)
            if len(page['text']) > chars_left:
                break  # Never cut source text and claim an incomplete line is visible.
            result['reads'].append({**page, 'purpose': purpose})
            lines_left -= count
            chars_left -= len(page['text'])

    for location in locations:
        tree = python_tree(executor, location['path']) if location['path'].endswith('.py') else None
        include(location['path'], implementation_ranges(tree, location['line']), 'supplied implementation and static neighbors')
    for runner in RUNNERS:
        if runner not in names:
            continue
        result['test_entrypoints'].append(runner)
        tree = python_tree(executor, runner)
        total = len(executor.source_lines(runner))
        setup = runner_setup_ranges(tree)
        spans = ([[0, min(60, total)], *setup, [max(0, total - 60), total]] if setup else
                 [[0, min(80, total)], [max(0, total - 120), total]])
        include(runner, merged(spans), 'native test entrypoint and configuration loading')
        for config in referenced_settings(tree, runner, names):
            include(config, [[0, 80]], 'literal runner settings default')
    for config in (name for name in CONFIGS if name in names):
        text = executor.source_bytes(config).decode('utf-8', errors='replace')
        match = re.search(r'(?mi)^\[[^\n]*(?:pytest|testenv)[^\n]*\]', text)
        start = text[:match.start()].count('\n') if match else 0
        if result['test_entrypoints'] and not match:
            continue
        include(config, [[start, start + 45]], 'project test configuration')
        if len([r for r in selected.values() if r == 'project test configuration']) >= 2:
            break
    for suite in suites:
        for filename, count in (('tests.py', 55), ('models.py', 180), ('conftest.py', 65)):
            include(suite + '/' + filename, [[0, count]], 'nearby native tests or fixtures')
    return result
