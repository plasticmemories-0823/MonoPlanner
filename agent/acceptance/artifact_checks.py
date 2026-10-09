"""Frozen raw-output checks. Initial consumer: numeric C expressions, not arbitrary artifacts."""

import ast
import json
import math
from pathlib import Path
import re
from agent.acceptance.c_driver import DRIVER_SOURCE




def required_consumer(request):
    """Conservative explicit C-output cues, not a general task classifier."""
    return "c_expression" if re.search(
        r"\bC(?:99|11)?\s+(?:code|expression|printer)\b|\b(?:generate|generated|printing)\s+C\b"
        r"|\bNot supported in C\b|生成\s*C\s*代码|C\s*代码生成", request) else None


def requested_calls(request):
    """Keep an explicit reproducible call chain tied to the request, not a model plan."""
    fallback = ()
    for match in re.finditer(r'\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(', request):
        fragment = request[match.start():match.start() + 2000]
        for end, char in enumerate(fragment):
            if char != ')':
                continue
            try:
                call = ast.parse(fragment[:end + 1], mode='eval').body
            except SyntaxError:
                continue
            if not isinstance(call, ast.Call):
                break
            names = tuple(sorted({n.func.id if isinstance(n.func, ast.Name) else n.func.attr
                for n in ast.walk(call) if isinstance(n, ast.Call)
                and isinstance(n.func, (ast.Name, ast.Attribute))} - {'print', 'repr', 'str'}))
            fallback = fallback or names
            if len(names) > 1:
                return names
            break
    return fallback


def _producer_calls(check):
    aliases = {a.asname or a.name: a.name for n in ast.parse(check['imports']).body for a in n.names}
    assignments = {t.id: n.value for n in ast.parse(check['setup']).body
                   for t in ast.walk(n.targets[0]) if isinstance(t, ast.Name)}
    seen, called = set(), set()
    def visit(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                called.add(aliases.get(n.func.id, n.func.id))
            if isinstance(n, ast.Name) and n.id in assignments and n.id not in seen:
                seen.add(n.id)
                visit(assignments[n.id])
    visit(ast.parse(check['expression'], mode='eval').body)
    return called


def load_artifact_checks(code, required=None, required_calls=()):
    declarations = [n for n in ast.parse(code).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "ARTIFACT_CHECKS" for t in n.targets)]
    if not declarations:
        if required:
            raise ValueError("本任务需要原始 C 产物验收；缺少顶格 ARTIFACT_CHECKS 字面量列表。")
        return []
    if len(declarations) != 1:
        raise ValueError("ARTIFACT_CHECKS 只能声明一次。")
    try:
        checks = ast.literal_eval(declarations[0].value)
    except (ValueError, TypeError) as exc:
        raise ValueError("ARTIFACT_CHECKS 必须是字面量，不能动态构造。") from exc
    if not isinstance(checks, list) or not 1 <= len(checks) <= 4:
        raise ValueError("ARTIFACT_CHECKS 必须包含 1 到 4 项。")
    seen = set()
    for c in checks:
        if not isinstance(c, dict):
            raise ValueError("Each artifact check must be a dictionary.")
        fields = {"id", "kind", "imports", "setup", "expression", "variables", "cases"}
        if c.get("kind") == "c_statement":
            fields.add("result_variable")
        if set(c) != fields:
            raise ValueError("Expected artifact fields; c_statement additionally requires result_variable.")
        if not isinstance(c['id'], str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,47}", c['id']) or c['id'] in seen:
            raise ValueError("产物检查 id 必须是唯一的短标识符。")
        seen.add(c['id'])
        if c['kind'] not in ('c_expression', 'c_statement'):
            raise ValueError("Only c_expression and c_statement consumers are supported.")
        for key, limit in (("imports", 1500), ("setup", 2000), ("expression", 1000)):
            if not isinstance(c[key], str) or len(c[key]) > limit:
                raise ValueError(f"产物检查 {key} 必须是至多 {limit} 字符的文本。")
        imported = set()
        try:
            imports = ast.parse(c['imports']).body
            setup = ast.parse(c['setup']).body
            call = ast.parse(c['expression'], mode='eval').body
        except SyntaxError as exc:
            raise ValueError(f"产物检查配置语法错误：行 {exc.lineno} 列 {exc.offset}，{exc.msg}") from exc
        for n in imports:
            if not isinstance(n, ast.ImportFrom) or n.level or not n.module or any(a.name == '*' for a in n.names):
                raise ValueError("imports 仅接受绝对 from 模块 import 名称，不允许通配符。")
            imported.update(a.asname or a.name for a in n.names)
        # Permit construction of input expressions, not definitions or changes to producers.
        allowed = (ast.Expression, ast.Name, ast.Load, ast.Constant, ast.Call, ast.keyword,
                   ast.List, ast.Tuple, ast.Dict, ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop,
                   ast.Compare, ast.cmpop, ast.BoolOp, ast.boolop, ast.Attribute, ast.Subscript, ast.Slice)
        for n in setup:
            if not isinstance(n, ast.Assign) or len(n.targets) != 1:
                raise ValueError("setup 只允许构造输入的变量赋值。")
            targets = list(ast.walk(n.targets[0]))
            if (any(not isinstance(p, (ast.Name, ast.Tuple, ast.List, ast.Store)) for p in targets)
                    or any(p.id in imported or p.id.startswith('_') for p in targets if isinstance(p, ast.Name))
                    or any(not isinstance(p, allowed) for p in ast.walk(n.value))):
                raise ValueError("setup 只允许构造输入的变量赋值，不能改写导入接口或定义转换器。")
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id not in imported
                or any(not isinstance(p, allowed) for p in ast.walk(call))):
            raise ValueError("expression 必须直接调用已导入的仓库接口，不能转换其返回值。")
        names = c['variables']
        if (not isinstance(names, list) or len(names) > 8
                or any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,30}', v) for v in names)
                or len(set(names)) != len(names)):
            raise ValueError("variables 必须是至多 8 个不同的 C 变量名。")
        if c['kind'] == 'c_statement':
            output = c['result_variable']
            reserved = set('auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while NAN INFINITY main agent_value printf'.split())
            if (not isinstance(output, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,30}', output)
                    or output in names or output in reserved):
                raise ValueError('result_variable must be a distinct, nonreserved C identifier.')
        cases = c['cases']
        if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
            raise ValueError("cases 必须包含 1 到 8 个数值检查。")
        for case in cases:
            if (not isinstance(case, dict) or set(case) != {'values', 'expected'}
                    or not isinstance(case['values'], list) or len(case['values']) != len(names)
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           for v in [*case['values'], case['expected']])):
                raise ValueError("case 需要与 variables 对齐的有限数值 values 和独立数值 expected。")
    if required_calls and not any(set(required_calls).issubset(_producer_calls(c)) for c in checks):
        raise ValueError('至少一项原始产物检查必须覆盖原需求中的调用链：' + '、'.join(required_calls) +
                         '。只编译相邻函数或把目标接口写在未使用的 setup 变量里不算覆盖。')
    return checks


# This script runs in the task's Python (including Python 3.9 images).
# It never loads the model-written test module; its frozen declaration is data.
_RUNNER = DRIVER_SOURCE + r'''
import ast, inspect, json, math, shutil, subprocess, sys, traceback
from pathlib import Path
config, output, source = map(Path, sys.argv[1:])
sys.path.insert(0, str(source))
checks = json.loads(config.read_text(encoding="utf-8"))
rows = []
compiler = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
for c in checks:
    prefix = 'artifact:' + c['id']
    current = [{'id':prefix + ':compile', 'status':'not_run', 'detail':''}]
    current += [{'id':prefix + ':case:' + str(i), 'status':'not_run', 'detail':''}
                for i in range(len(c['cases']))]
    folder = output.parent / c['id']
    folder.mkdir(exist_ok=True)
    try:
        if not compiler:
            raise RuntimeError('目标环境缺少 C 编译器，不能判定产物通过。')
        scope = {}
        exec(c['imports'], scope)
        target = scope[ast.parse(c['expression'], mode='eval').body.func.id]
        definition = inspect.getsourcefile(target)
        if not definition or not Path(definition).resolve().is_relative_to(source.resolve()):
            raise ValueError('生成接口必须来自当前仓库，不能用替代函数产生答案。')
        exec(c['setup'], scope)
        raw = eval(c['expression'], scope)
        if not isinstance(raw, str) or not raw.strip() or len(raw) > 65536:
            raise ValueError('接口需要返回非空原始 C 文本，最多 65536 字符。')
        (folder / 'raw.txt').write_bytes(raw.encode('utf-8'))
        generated = build_c_driver(c, raw)
        path, exe = folder / 'generated.c', folder / 'generated.exe'
        path.write_bytes(generated.encode('utf-8'))
        args = [compiler, '-std=c99', '-Werror=implicit-function-declaration', str(path), '-lm', '-o', str(exe)]
        compiled = subprocess.run(args, capture_output=True, text=True, errors='replace', timeout=30)
        (folder / 'compile.stdout.log').write_text(compiled.stdout, encoding='utf-8')
        (folder / 'compile.stderr.log').write_text(compiled.stderr, encoding='utf-8')
        (folder / 'compile.json').write_text(json.dumps({'argv':args,'exit_code':compiled.returncode}), encoding='utf-8')
        if compiled.returncode:
            current[0].update(status='failed', detail='原始产物编译失败，不能通过转换后验收。\n' + compiled.stderr[-4000:])
        else:
            current[0]['status'] = 'passed'
            executed = subprocess.run([str(exe)], capture_output=True, text=True, errors='replace', timeout=10)
            (folder / 'execution.json').write_text(json.dumps({'exit_code':executed.returncode,
                'stdout':executed.stdout,'stderr':executed.stderr}), encoding='utf-8')
            values = executed.stdout.splitlines()
            for i, case in enumerate(c['cases']):
                try:
                    actual = float(values[i]) if not executed.returncode else float('nan')
                except (IndexError, ValueError):
                    actual = float('nan')
                good = len(values) == len(c['cases']) and math.isfinite(actual) and math.isclose(actual, case['expected'], rel_tol=1e-9, abs_tol=1e-9)
                current[i + 1].update(status='passed' if good else 'failed', values=case['values'],
                    expected=case['expected'], actual=actual if math.isfinite(actual) else None,
                    exit_code=executed.returncode,
                    detail='' if good else '输入 ' + str(case['values']) + '；预期 ' + str(case['expected']) +
                        '；实际 ' + str(actual) + '；退出码 ' + str(executed.returncode))
    except subprocess.TimeoutExpired as exc:
        current[0].update(status='error', detail='真实消费者超时，验收未完成：' + str(exc))
    except Exception:
        current[0].update(status='error', detail=traceback.format_exc()[-4000:])
    rows.extend(current)
output.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
'''


def run_artifact_checks(checks, environment, directory, timeout):
    """Run the frozen producer and consumer separately from model-written tests."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config, output = directory / 'checks.json', directory / 'results.json'
    config.write_text(json.dumps(checks, ensure_ascii=False), encoding='utf-8')
    output.unlink(missing_ok=True)
    mapped = environment.host_to_container
    command = [environment.python_executable, '-B', '-X', 'utf8', '-c', _RUNNER,
               mapped(config), mapped(output), mapped(environment.repo_root)]
    outcome = environment.run_process(command, stdout_path=directory/'stdout.log',
                                     stderr_path=directory/'stderr.log', timeout=timeout, shell=False)
    if outcome.get('timed_out') or outcome.get('error') or outcome['exit_code'] != 0:
        raise RuntimeError('原始产物检查进程未正常完成：' + str(outcome))
    rows = json.loads(output.read_text(encoding='utf-8'))
    expected = {f"artifact:{c['id']}:compile" for c in checks} | {
        f"artifact:{c['id']}:case:{i}" for c in checks for i in range(len(c['cases']))}
    if (not isinstance(rows, list) or len(rows) != len(expected) or
            {r.get('id') for r in rows} != expected or any(r.get('status') not in
                {'passed', 'failed', 'error', 'not_run'} or not isinstance(r.get('detail'), str) for r in rows)):
        raise RuntimeError('原始产物检查没有返回完整的固定检查项。')
    return rows
