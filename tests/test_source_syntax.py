"""冻结源码准备使用的静态解析；不依赖已移除的实时导航器。"""

from agent.source.syntax import _parse


def test_aliases_relative_imports_and_call_locations():
    entry = _parse(
        b'from .arithmetic import add as add_pair\n\n'
        b'def total(values):\n    return add_pair(*values)\n',
        'src/tinycalc/api.py',
    )
    assert entry['imports'][0]['binding'] == 'add_pair'
    assert entry['imports'][0]['target'] == 'tinycalc.arithmetic.add'
    assert entry['calls'][0]['targets'] == ['tinycalc.arithmetic.add']
    assert entry['calls'][0]['line'] == 4
    symbol = entry['symbols'][0]
    assert symbol['signature'] == 'def total(values)'
    assert (symbol['line'], symbol['end_line']) == (3, 4)


def test_class_async_signature_and_doc_without_execution(tmp_path):
    marker = tmp_path / 'SHOULD_NOT_EXIST'
    source = (
        f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
        'class Client:\n    async def send(self, text: str) -> int:\n'
        '        """Send one text."""\n        return len(text)\n'
    )
    entry = _parse(source.encode('utf-8'), 'client.py')
    assert not marker.exists()
    method = next(s for s in entry['symbols'] if s['name'] == 'Client.send')
    assert method['signature'] == 'async def send(self, text: str) -> int'
    assert method['doc'] == 'Send one text.'
    assert (method['line'], method['end_line']) == (4, 6)
