"""只解析仓库源码中的 Python 定义和静态继承，不导入或执行待验收项目。"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import stat
import types


OBJECT = ('builtins', ('object',))
ABC = ('abc', ('ABC',))


def dotted(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + (node.attr,)
    raise ValueError('API inheritance requires statically named repository bases')


def binding(body, name):
    """赋值和条件定义也会遮蔽同名方法，不能跳过它们去找父类。"""
    found = []
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                found.append((node, None))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                local = alias.asname or (alias.name.split('.')[0]
                                        if isinstance(node, ast.Import) else alias.name)
                if local in (name, '*'):
                    found.append((node, alias))
        elif any(
            (isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del))
             and item.id == name)
            or (isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == name)
            or (isinstance(item, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or alias.name.split('.')[0]) in (name, '*')
                for alias in item.names))
            for item in ast.walk(node)
        ):
            found.append((node, None))
    if len(found) > 1:
        raise ValueError('API symbol has ambiguous or conditional source bindings: ' + name)
    return found[0] if found else None


class SourceResolver:
    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        self.modules = {}
        self.mros = {OBJECT: [OBJECT], ABC: [ABC, OBJECT]}

    def path(self, module):
        parts = module.split('.')
        if not parts or not all(part.isidentifier() for part in parts):
            raise ValueError('API module must be a dotted source name')
        path = self.repo.joinpath(*parts)
        source = path / '__init__.py' if (path / '__init__.py').is_file() else path.with_suffix('.py')
        current = self.repo
        for part in source.relative_to(self.repo).parts:
            current /= part
            if current.exists() or current.is_symlink():
                info = current.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('API source links and reparse points are unsupported')
        return source

    def module(self, name):
        if name not in self.modules:
            source = self.path(name)
            if not source.is_file():
                raise ValueError('API source is outside the supported repository modules: ' + name)
            raw = source.read_bytes()
            tree = ast.parse(raw, filename=str(source))
            compiled = compile(tree, str(source), 'exec', dont_inherit=True)
            self.modules[name] = (source, tree, compiled, hashlib.sha256(raw).hexdigest())
        return self.modules[name]

    def imported_name(self, module, node, alias):
        if isinstance(node, ast.Import):
            return alias.name if alias.asname else alias.name.split('.')[0]
        package = module if self.module(module)[0].name == '__init__.py' else module.rpartition('.')[0]
        target = node.module or ''
        if node.level:
            try:
                target = importlib.util.resolve_name('.' * node.level + target, package)
            except (ImportError, ValueError) as exc:
                raise ValueError('API source import cannot be resolved: ' + str(exc)) from exc
        return target + '.' + alias.name

    def qualified(self, name, trail):
        parts = tuple(name.split('.'))
        # 标准 ABC 和 object 的线性继承固定；不把任意外部基类当作空类跳过。
        if name in ('abc.ABC', 'abc.ABCMeta', 'abc.abstractmethod', 'builtins.object'):
            if not self.path(parts[0]).is_file():
                return parts[0], parts[1:]
        for size in range(len(parts) - 1, 0, -1):
            module = '.'.join(parts[:size])
            if self.path(module).is_file():
                return self.lookup(module, parts[size:], trail)
        raise ValueError('API base or imported symbol has no static repository definition: ' + name)

    def lookup(self, module, names, trail=()):
        identity = (module, tuple(names))
        if identity in trail or len(trail) >= 64:
            raise ValueError('API source bindings are cyclic or too deep')
        trail = trail + (identity,)
        body = self.module(module)[1].body
        prefix = ()
        for index, name in enumerate(names):
            match = binding(body, name)
            if match is None:
                if prefix:
                    for parent in self.mro((module, prefix))[1:]:
                        if parent in (OBJECT, ABC):
                            continue
                        parent_node = self.node(parent)
                        if binding(parent_node.body, name) is not None:
                            return self.lookup(parent[0], parent[1] + tuple(names[index:]), trail)
                raise ValueError('API symbol must identify one original Python function or method: ' + name)
            node, alias = match
            if isinstance(node, (ast.Import, ast.ImportFrom)) and not prefix:
                if alias.name == '*':
                    raise ValueError('API wildcard source bindings are unsupported')
                target = self.imported_name(module, node, alias)
                suffix = tuple(names[index + 1:])
                return self.qualified('.'.join((target,) + suffix), trail)
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ValueError('API symbol is shadowed by an unsupported source binding: ' + name)
            prefix += (name,)
            if index + 1 < len(names) and not isinstance(node, ast.ClassDef):
                raise ValueError('API nested functions are not public class methods')
            body = node.body
        return module, prefix

    def node(self, identity):
        body = self.module(identity[0])[1].body
        for name in identity[1]:
            match = binding(body, name)
            if match is None:
                raise ValueError('API definition disappeared from its source')
            node = match[0]
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ValueError('API definition is not a function or class')
            body = node.body
        return node

    def reference(self, module, scope, expression):
        names = dotted(expression)
        body = self.module(module)[1].body
        if names == ('object',) and binding(body, 'object') is None:
            return OBJECT
        # 嵌套类定义里的基类名先在外层类体查找，再查模块。
        if scope and binding(self.node((module, scope)).body, names[0]) is not None:
            return self.lookup(module, scope + names)
        return self.lookup(module, names)

    def mro(self, identity, pending=()):
        if identity in self.mros:
            return self.mros[identity]
        if identity in pending or len(pending) >= 64:
            raise ValueError('API inheritance is cyclic or too deep')
        node = self.node(identity)
        if not isinstance(node, ast.ClassDef) or node.decorator_list:
            raise ValueError('API inherited methods require an undecorated source class')
        for keyword in node.keywords:
            if (keyword.arg != 'metaclass' or self.reference(
                    identity[0], identity[1][:-1], keyword.value) != ('abc', ('ABCMeta',))):
                raise ValueError('API custom metaclasses cannot supply static inheritance evidence')
        bases = [self.reference(identity[0], identity[1][:-1], base) for base in node.bases]
        bases = bases or [OBJECT]
        if len(set(bases)) != len(bases):
            raise ValueError('API inheritance contains duplicate bases')
        orders = [list(self.mro(base, pending + (identity,))) for base in bases] + [list(bases)]
        result = [identity]
        # C3 保留多继承中的覆盖顺序，不能按深度优先找到第一个同名定义就放行。
        while any(orders):
            orders = [order for order in orders if order]
            head = next((order[0] for order in orders
                         if not any(order[0] in other[1:] for other in orders)), None)
            if head is None:
                raise ValueError('API inheritance has no consistent Python method order')
            result.append(head)
            for order in orders:
                if order[0] == head:
                    order.pop(0)
        self.mros[identity] = result
        return result

    def abstract_placeholder(self, identity, node):
        abstract = False
        for decorator in node.decorator_list:
            try:
                abstract |= self.reference(identity[0], (), decorator) == ('abc', ('abstractmethod',))
            except ValueError:
                continue
        if not abstract:
            return False
        for item in node.body:
            if isinstance(item, ast.Pass):
                continue
            if isinstance(item, ast.Expr):
                value = getattr(item.value, 'value', getattr(item.value, 's', None))
                if (isinstance(value, str) or value is Ellipsis
                        or type(item.value).__name__ == 'Ellipsis'):
                    continue
            if isinstance(item, ast.Return) and (item.value is None or (
                    type(item.value).__name__ in ('Constant', 'NameConstant')
                    and item.value.value is None)):
                continue
            if isinstance(item, ast.Raise):
                error = item.exc.func if isinstance(item.exc, ast.Call) else item.exc
                if isinstance(error, ast.Name) and error.id == 'NotImplementedError':
                    continue
            return False
        return True

    def resolve(self, entry):
        module, symbol = entry.split(':')
        identity = self.lookup(module, tuple(symbol.split('.')))
        node = self.node(identity)
        if not isinstance(node, ast.FunctionDef):
            raise ValueError('API evidence requires a synchronous Python function or method')
        if self.abstract_placeholder(identity, node):
            raise ValueError('An abstract placeholder cannot supply API consumption evidence; '
                             'bind the concrete implementation exercised by the required tests')
        source, _, compiled, source_hash = self.module(identity[0])
        for name in identity[1]:
            codes = [item for item in compiled.co_consts
                     if isinstance(item, types.CodeType) and item.co_name == name]
            if len(codes) != 1:
                raise ValueError('API compiled definition is ambiguous')
            compiled = codes[0]
        return {
            'source': str(source), 'source_sha256': source_hash,
            'source_line': min([node.lineno] + [item.lineno for item in node.decorator_list]),
            'symbol': symbol, 'resolved_api_entry': identity[0] + ':' + '.'.join(identity[1]),
            'source_inventory': {path.relative_to(self.repo).as_posix(): digest
                                 for path, _, _, digest in self.modules.values()},
        }, compiled


def resolve_api(repo, entry):
    return SourceResolver(repo).resolve(entry)
