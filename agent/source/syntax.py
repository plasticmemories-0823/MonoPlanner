"""解析地图实际使用的 Python 定义与导入；不执行源码或收集无消费者的调用图。"""

import ast
from pathlib import PurePosixPath


MAX_FILE_BYTES = 512 * 1024


def _module(path):
    parts = list(PurePosixPath(path).with_suffix('').parts)
    if parts and parts[0] == 'src':
        parts.pop(0)
    if parts and parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


def _parse(data, path):
    tree = ast.parse(data, filename=path)
    module = _module(path)
    symbols, imports = [], []
    pending = [(tree, '')]
    while pending:
        node, prefix = pending.pop()
        scope = prefix
        location = {'line': getattr(node, 'lineno', 1),
                    'end_line': getattr(node, 'end_lineno', 1)}
        identity = f'{location["line"]}:{getattr(node, "col_offset", 0)}'
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + node.name
            symbols.append({'id': 'symbol:' + identity, 'name': name,
                            'kind': 'class' if isinstance(node, ast.ClassDef) else 'function',
                            **location})
            scope = name + '.'
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            base = ''
            if isinstance(node, ast.ImportFrom):
                package = (module.split('.') if PurePosixPath(path).name == '__init__.py'
                           else module.split('.')[:-1])
                parent = package[:max(0, len(package) - node.level + 1)] if node.level else []
                base = '.'.join([*parent, *([node.module] if node.module else [])])
            for index, alias in enumerate(node.names):
                imports.append({'id': f'import:{identity}:{index}',
                                'target': '.'.join(filter(None, (base, alias.name))), **location})
        pending.extend((child, scope) for child in reversed(list(ast.iter_child_nodes(node))))
    return {'symbols': symbols, 'imports': imports,
            'total_lines': len(data.decode('utf-8-sig').splitlines()),
            'parser': {'status': 'parsed', 'complete': True}}
