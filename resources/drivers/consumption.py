"""在现有 unittest 执行内记录库调用和原始文件读取，不再次执行任务。"""
import ast
import hashlib
from pathlib import Path
import sys
import types


MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inside(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class Witness:
    def __init__(self, checks, repo, test_path):
        self.repo = Path(repo).resolve()
        self.test_path = Path(test_path).resolve()
        self.active = None
        self.rows = []
        self.matches = {}
        self.profile_cache = {}
        self.expected_code = {}
        self.previous_profile = None
        self.profile_callback = self.observe
        for check in checks:
            if check['consumer'] != 'python_tests':
                continue
            row = {
                'id': check['id'], 'consumer': 'python_tests',
                'artifact_type': check['artifact_type'], 'target_runtime': 'python',
                'test_ids': check['test_ids'],
                'calls': {name: 0 for name in check['test_ids']},
                'status': 'error', 'error_kind': 'test',
            }
            if check['artifact_type'] == 'python_api':
                self.prepare_api(check, row)
            else:
                self.prepare_artifact(check, row)
            self.rows.append(row)

    def prepare_api(self, check, row):
        module, symbol = check['api_entry'].split(':')
        path = self.repo.joinpath(*module.split('.'))
        source = path / '__init__.py' if (path / '__init__.py').is_file() else path.with_suffix('.py')
        row.update(api_entry=check['api_entry'], source=str(source.resolve()),
                   source_sha256=None, syntax_checked=False)
        if not inside(source, self.repo) or not source.is_file():
            row.update(error_kind='environment', reason='The declared repository API source is missing')
            return
        raw = source.read_bytes()
        row['source_sha256'] = hashlib.sha256(raw).hexdigest()
        try:
            tree = ast.parse(raw, filename=str(source.resolve()))
            compiled = compile(tree, str(source.resolve()), 'exec', dont_inherit=True)
            body = tree.body
            node = None
            for name in symbol.split('.'):
                matches = [item for item in body if isinstance(
                    item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name]
                if len(matches) != 1:
                    raise ValueError('API symbol must identify one original Python function or method')
                node = matches[0]
                body = node.body
            if not isinstance(node, ast.FunctionDef):
                raise ValueError('API evidence requires a synchronous Python function or method')
            for name in symbol.split('.'):
                codes = [item for item in compiled.co_consts
                         if isinstance(item, types.CodeType) and item.co_name == name]
                if len(codes) != 1:
                    raise ValueError('API compiled definition is ambiguous')
                compiled = codes[0]
            line = min([node.lineno] + [item.lineno for item in node.decorator_list])
            key = (str(source.resolve()), node.name, line)
            self.matches.setdefault(key, []).append(row)
            self.expected_code[row['id']] = compiled
            row.update(syntax_checked=True, source_line=line, symbol=symbol,
                       compiled_definition_checked=True)
        except (SyntaxError, ValueError) as exc:
            row.update(error_kind='source', reason=str(exc))

    def prepare_artifact(self, check, row):
        path = self.repo / check['entry_file']
        row.update(artifact_path=check['entry_file'], artifact_sha256=None,
                   bytes_read=0, missing_at_start=not path.exists())
        if not inside(path, self.repo) or path.is_symlink():
            row.update(error_kind='integrity', reason='Artifact path leaves the original repository')
        elif path.exists():
            if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
                row.update(error_kind='environment', reason='Artifact is not a bounded regular file')
            else:
                row['artifact_sha256'] = digest(path)

    def install(self):
        module = types.ModuleType('acceptance_artifacts')
        module.read_artifact = self.read_artifact
        sys.modules['acceptance_artifacts'] = module

    def start(self, test):
        self.active = '.'.join(test.id().split('.')[-2:])
        self.previous_profile = sys.getprofile()
        if self.matches:
            sys.setprofile(self.profile_callback)

    def stop(self):
        if self.active is None:
            return
        if self.matches:
            if sys.getprofile() is not self.profile_callback:
                for row in self.rows:
                    row.update(error_kind='integrity', reason='API observation was replaced during testing')
            sys.setprofile(self.previous_profile)
        self.active = None

    def in_test_body(self, frame):
        if not self.active:
            return False
        method = self.active.split('.')[-1]
        while frame:
            code = frame.f_code
            if code.co_name == method and code.co_filename == str(self.test_path):
                return True
            frame = frame.f_back
        return False

    def observe(self, frame, event, arg):
        if event != 'call' or not self.active:
            return
        code = frame.f_code
        identity = id(code)
        if identity not in self.profile_cache:
            self.profile_cache[identity] = (code, self.matches.get(
                (code.co_filename, code.co_name, code.co_firstlineno), ()))
        rows = self.profile_cache[identity][1]
        if rows and self.in_test_body(frame):
            for row in rows:
                if (self.active in row['calls']
                        and code == self.expected_code[row['id']]):
                    row['calls'][self.active] += 1

    def read_artifact(self, relative):
        rows = [row for row in self.rows if row.get('artifact_path') == relative
                and self.active in row['calls']]
        if not rows or not self.in_test_body(sys._getframe(1)):
            raise ValueError('Read the declared artifact from its bound unittest method')
        path = self.repo / relative
        for row in rows:
            row['calls'][self.active] += 1
        for row in rows:
            if row.get('error_kind') in {'integrity', 'environment'}:
                raise ValueError(row['reason'])
            if row['missing_at_start']:
                if path.exists():
                    row.update(error_kind='integrity', reason='Acceptance created the missing deliverable')
                    raise ValueError(row['reason'])
                raise AssertionError('Required delivered artifact is missing: ' + relative)
        if not path.is_file() or path.is_symlink() or not inside(path, self.repo):
            raise ValueError('The original artifact path changed')
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        for row in rows:
            if actual != row['artifact_sha256'] or digest(path) != actual:
                row.update(error_kind='integrity', reason='Delivered artifact changed during acceptance')
                raise ValueError(row['reason'])
            row['bytes_read'] += len(raw)
        return raw

    def finish(self, test_rows):
        self.stop()
        tests = {'.'.join(row['id'].split('.')[-2:]): row for row in test_rows}
        for row in self.rows:
            if row.get('error_kind') in {'integrity', 'environment'}:
                continue
            if not all(name in tests and tests[name]['status'] in {'passed', 'failed', 'error'}
                       and count > 0 for name, count in row['calls'].items()):
                row.update(error_kind='test', reason='Required API or artifact was not consumed by each bound test')
                continue
            if row['artifact_type'] == 'python_api':
                source = Path(row['source'])
                if not row['syntax_checked'] or digest(source) != row['source_sha256']:
                    row.update(error_kind='integrity', reason='Consumed API source changed')
                    continue
            else:
                path = self.repo / row['artifact_path']
                if row['missing_at_start'] and not path.exists():
                    if any(tests[name]['status'] == 'passed' for name in row['test_ids']):
                        row.update(error_kind='test', reason='A test accepted the missing required artifact')
                    else:
                        row.update(status='failed', error_kind='source', reason='Required artifact is absent')
                    continue
                if not path.is_file() or path.is_symlink() or digest(path) != row['artifact_sha256']:
                    row.update(error_kind='integrity', reason='Consumed artifact changed')
                    continue
            row.update(status='passed', error_kind=None)
        return {'checks': self.rows}
