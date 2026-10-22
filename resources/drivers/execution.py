"""在既有隔离后端执行真实入口和显式选择的产物使用方式。"""

import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            value.update(block)
    return value.hexdigest()


def contained(path, root):
    try:
        Path(path).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def tail(path):
    with Path(path).open('rb') as stream:
        stream.seek(max(0, Path(path).stat().st_size - 4000))
        return stream.read().decode('utf-8', 'replace')


def compile_artifact(argv, source, folder, row, remaining):
    """固定编译步骤先于运行，记录实际编译原件而不是模型声称的成功。"""
    digest = sha(source)
    stdout, stderr = folder / 'compile.stdout', folder / 'compile.stderr'
    with stdout.open('wb') as out, stderr.open('wb') as err:
        result = subprocess.run(argv, stdout=out, stderr=err, timeout=max(.1, remaining()))
    row.update(compile_argv=argv, compile_exit_code=result.returncode,
               compiled_source_sha256=digest)
    if sha(source) != digest:
        row['error_kind'] = 'integrity'
        raise ValueError('Compiled source changed during compilation')
    if result.returncode:
        row.update(status='failed', error_kind='source', detail=tail(stderr))
        return False
    return True


def run_command(check, repo, folder, row, remaining):
    """由固定驱动拼接入口参数，入口不能被挪到解释器代码参数之后。"""
    entry = repo / check['entry_file']
    if not contained(entry, repo):
        raise ValueError('Command entry escapes the repository')
    if not entry.is_file():
        row.update(status='failed', error_kind='source', detail='Required repository entry does not exist')
        return row
    row.update(producer_source=str(entry.resolve()), producer_source_sha256=sha(entry))
    interpreter = check['interpreter']
    if interpreter == 'python':
        prefix = [sys.executable, '-B', '-X', 'utf8']
        compile_argv = prefix + [
            '-c', 'import py_compile,sys; py_compile.compile(sys.argv[1], '
            'cfile=sys.argv[2], doraise=True)', str(entry.resolve()), str(folder / 'entry.pyc')]
    else:
        programs = {'shell': ('sh', '-n'), 'node': ('node', '--check')}
        if interpreter not in programs:
            raise ValueError('Unsupported compiler/runtime; direct executables cannot bypass compilation')
        name, syntax_option = programs[interpreter]
        program = shutil.which(name)
        if not program:
            row['error_kind'] = 'environment'
            raise RuntimeError('Declared interpreter is unavailable in the fixed environment')
        prefix = [program]
        compile_argv = prefix + [syntax_option, str(entry.resolve())]
    row['error_kind'] = 'environment'
    if not compile_artifact(compile_argv, entry, folder, row, remaining):
        return row
    if sha(entry) != row['producer_source_sha256']:
        row['error_kind'] = 'integrity'
        raise ValueError('Command entry changed before execution')
    argv = prefix + [str(entry.resolve())] + check['argv']
    stdout, stderr = folder / 'command.stdout', folder / 'command.stderr'
    row.update(error_kind='environment', argv=argv)
    with stdout.open('wb') as out, stderr.open('wb') as err:
        result = subprocess.run(argv, cwd=str(repo), stdout=out, stderr=err,
                                timeout=max(.1, remaining()))
    row.update(producer_called=True, exit_code=result.returncode,
               producer_result_sha256=sha(stdout), stderr_sha256=sha(stderr),
               stdout=tail(stdout), stderr=tail(stderr), error_kind=None,
               status='passed' if result.returncode == 0 else 'failed')
    if sha(entry) != row['producer_source_sha256']:
        row['error_kind'] = 'integrity'
        raise ValueError('Command entry changed during execution')
    return row


def c_source(check, raw):
    parameters = ', '.join('double ' + name for name in check['variables']) or 'void'
    if check['consumer'] == 'c_expression':
        body = 'return (\n' + raw + '\n);'
    else:
        result = check['result_variable']
        body = 'double ' + result + ' = NAN;\n' + raw + '\nreturn ' + result + ';'
    arguments = ', '.join('strtod(argv[' + str(index + 1) + '], 0)'
                          for index in range(len(check['variables'])))
    return ('#include <math.h>\n#include <stdio.h>\n#include <stdlib.h>\n'
            + 'double managed_value(' + parameters + ') {\n' + body + '\n}\n'
            + 'int main(int argc, char **argv) {\n'
            + 'if (argc != ' + str(len(check['variables']) + 1) + ') return 2;\n'
            + 'printf("%.17g\\n", managed_value(' + arguments + '));\nreturn 0;\n}\n')


def execute(check, repo, folder, remaining):
    started = time.monotonic()
    row = {'id': check['id'], 'consumer': check['consumer'], 'status': 'error',
           'artifact_type': check.get('artifact_type'),
           'target_runtime': check.get('interpreter') if check['consumer'] == 'command' else 'c',
           'error_kind': 'setup', 'producer_called': False, 'runs': []}
    try:
        consumer = check['consumer']
        if consumer not in ('command', 'c_expression', 'c_statement'):
            raise ValueError('Every execution declaration requires compilation and consumption')
        expected_type = 'repository_entry' if consumer == 'command' else 'generated_code'
        if check.get('artifact_type') != expected_type:
            raise ValueError('Artifact type does not match the required consumer')
        if consumer != 'command' and not check.get('cases'):
            raise ValueError('Generated numeric code requires runtime cases after compilation')
        if remaining() <= 0:
            row['error_kind'] = 'timeout'
            raise RuntimeError('Managed execution time budget is exhausted')
        folder.mkdir(parents=True, exist_ok=False)
        if check['consumer'] == 'command':
            return run_command(check, repo, folder, row, remaining)
        namespace = {}
        exec(compile(check['imports'], '<managed-imports>', 'exec'), namespace)
        call = ast.parse(check['expression'], mode='eval').body
        producer = namespace[call.func.id]
        source = inspect.getsourcefile(producer)
        if not callable(producer) or not source or not contained(source, repo):
            raise ValueError('Producer must resolve to the actual repository source')
        row.update(producer_module=getattr(producer, '__module__', ''),
                   producer_name=getattr(producer, '__qualname__', call.func.id),
                   producer_source=str(Path(source).resolve()), producer_source_sha256=sha(source))
        exec(compile(check['setup'], '<managed-inputs>', 'exec'), namespace)
        if namespace[call.func.id] is not producer:
            raise ValueError('Producer binding was replaced during input preparation')
        args = [eval(compile(ast.Expression(arg), '<managed-argument>', 'eval'), namespace)
                for arg in call.args]
        kwargs = {item.arg: eval(compile(ast.Expression(item.value), '<managed-keyword>', 'eval'), namespace)
                  for item in call.keywords}
        row['producer_called'] = True
        row['error_kind'] = 'source'
        raw = producer(*args, **kwargs)
        if sha(source) != row['producer_source_sha256']:
            raise ValueError('Producer source changed during execution')
        row['error_kind'] = 'behavior'
        if not isinstance(raw, str) or not raw or len(raw.encode('utf-8')) > 1_048_576:
            raise ValueError('Numeric code consumer requires a nonempty bounded raw text artifact')
        artifact = folder / 'raw.txt'
        artifact.write_bytes(raw.encode('utf-8'))
        row['artifact_sha256'] = sha(artifact)
        source_file = folder / 'consumer.c'
        source_file.write_bytes(c_source(check, raw).encode('utf-8'))
        executable = folder / ('consumer.exe' if sys.platform == 'win32' else 'consumer')
        compiler = next((shutil.which(name) for name in ('cc', 'gcc', 'clang')
                         if shutil.which(name)), None)
        if compiler is None:
            row['error_kind'] = 'environment'
            raise RuntimeError('No compiler is available in the fixed execution environment')
        argv = [compiler, '-std=c99', '-Werror=implicit-function-declaration',
                str(source_file), '-lm', '-o', str(executable)]
        row['consumer_source_sha256'] = sha(source_file)
        if not compile_artifact(argv, source_file, folder, row, remaining):
            return row
        row.update(executable_name=executable.name, executable_sha256=sha(executable))
        for index, case in enumerate(check['cases']):
            result = subprocess.run([str(executable)] + [repr(v) for v in case['values']],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=max(.1, remaining()))
            output = result.stdout.decode('utf-8', 'replace')
            out_path, err_path = folder / ('run-%d.stdout' % index), folder / ('run-%d.stderr' % index)
            out_path.write_bytes(result.stdout)
            err_path.write_bytes(result.stderr)
            try:
                actual = float(output.strip())
                passed = (result.returncode == 0 and math.isfinite(actual)
                          and math.isclose(actual, case['expected'], rel_tol=1e-9, abs_tol=1e-12))
            except ValueError:
                actual, passed = None, False
            row['runs'].append({'values': case['values'], 'expected': case['expected'],
                                'actual': actual if actual is None or math.isfinite(actual) else None,
                                'exit_code': result.returncode, 'stdout': output[-4000:],
                                'stderr': result.stderr.decode('utf-8', 'replace')[-4000:],
                                'stdout_sha256': sha(out_path), 'stderr_sha256': sha(err_path),
                                'passed': passed})
        row.update(status='passed' if all(run['passed'] for run in row['runs']) else 'failed',
                   error_kind=None)
        if sha(artifact) != row['artifact_sha256'] or sha(source_file) != row['consumer_source_sha256']:
            raise ValueError('Raw artifact or consumer source changed during execution')
    except subprocess.TimeoutExpired:
        row.update(status='error', error_kind='timeout', detail='Managed execution exceeded its time budget')
    except Exception as error:
        row.update(status='error', detail=type(error).__name__ + ': ' + str(error),
                   traceback=traceback.format_exc()[-4000:])
    finally:
        row['seconds'] = time.monotonic() - started
    return row


def main():
    declaration, repo, output, budget = sys.argv[1:]
    repo = Path(repo).resolve()
    output = Path(output)
    sys.path.insert(0, str(repo))
    deadline = time.monotonic() + float(budget)
    checks = json.loads(Path(declaration).read_text(encoding='utf-8'))
    rows = []
    for check in checks:
        rows.append(execute(check, repo, output.parent / (output.stem + '-' + check['id']),
                            lambda: deadline - time.monotonic()))
    output.write_text(json.dumps({'checks': rows}, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
