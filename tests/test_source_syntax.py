"""冻结源码准备使用的静态解析；不依赖已移除的实时导航器。"""

from agent.source.syntax import _parse


def test_relative_imports_and_definition_locations():
    entry = _parse(
        b'from .arithmetic import add as add_pair\n\n'
        b'def total(values):\n    return add_pair(*values)\n',
        'src/tinycalc/api.py',
    )
    assert entry['imports'][0]['target'] == 'tinycalc.arithmetic.add'
    assert entry['imports'][0]['id'] == 'import:1:0:0'
    assert entry['imports'][0]['end_line'] == 1
    symbol = entry['symbols'][0]
    assert symbol['name'] == 'total'
    assert (symbol['line'], symbol['end_line']) == (3, 4)


def test_class_async_locations_without_execution(tmp_path):
    marker = tmp_path / 'SHOULD_NOT_EXIST'
    source = (
        f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
        'class Client:\n    async def send(self, text: str) -> int:\n'
        '        """Send one text."""\n        return len(text)\n'
    )
    entry = _parse(source.encode('utf-8'), 'client.py')
    assert not marker.exists()
    method = next(s for s in entry['symbols'] if s['name'] == 'Client.send')
    assert method['kind'] == 'function'
    assert (method['line'], method['end_line']) == (4, 6)


def test_parser_keeps_all_definitions_and_imports_including_conditional_scopes():
    raw = (''.join(f'import m{i}\n' for i in range(180))
           + 'if True:\n' + ''.join(f'    def f{i}(): pass\n' for i in range(180)))
    entry = _parse(raw.encode(), 'many.py')
    assert len(entry['imports']) == len(entry['symbols']) == 180
    assert entry['parser']['complete']
    assert len({node['id'] for node in entry['symbols']}) == 180


def test_parameter_text_does_not_inflate_navigation():
    short = _parse(b'def long(x):\n    pass\n', 'a.py')['symbols']
    raw = 'def long(' + ', '.join(f'argument_{i}=0' for i in range(80)) + '):\n    pass\n'
    assert _parse(raw.encode(), 'a.py')['symbols'] == short
