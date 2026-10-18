"""按已观察范围发现导航条目；只有追加到工具历史的条目才确认投递。"""

from copy import deepcopy
import hashlib
import json

from agent.source.syntax import _parse, _module, MAX_FILE_BYTES
from agent.source.paging import merged, read_pages


MAX_MAP_BYTES = 4000
_FIELDS = {'new_symbols': 5, 'new_dependencies': 3}


def _receipt():
    return {'ids': [], 'parser': False}


def _accept(receipts, updates, state):
    for update in updates:
        name = update['path']
        if name not in state or state[name]['sha256'] != update.get('sha256'):
            continue
        receipt = receipts.setdefault(name, _receipt())
        receipt['ids'] = sorted(set(receipt['ids']).union(
            node['id'] for field in ('new_symbols', 'new_dependencies')
            for node in update.get(field, [])
        ))
        receipt['parser'] |= 'parser' in update


class ExplorationMap:
    def __init__(self, executor, state, coverage=None):
        self.executor, self.state = executor, state
        self.reported = coverage if coverage is not None else {}
        self.parsed = {}
        self.version, self.modules = None, {}

    def sync(self):
        files = self.executor.inventory()
        if self.version != self.executor.evidence.current:
            for name in list(self.state):
                if files.get(name) != self.state[name]['sha256']:
                    del self.state[name]
            self.modules = {_module(name): name for name in files if name.endswith('.py')}
            self.version = self.executor.evidence.current
            self.parsed = {name: entry for name, entry in self.parsed.items()
                           if files.get(name) == entry['digest']}
        return files

    def entry(self, name, sha):
        cached = self.parsed.get(name)
        if cached and cached['digest'] == sha:
            return cached
        prepared = self.executor.prepared
        cached = (prepared['entries'].get(name)
                  if prepared and prepared['version'] == self.version else None)
        if cached and cached['digest'] == sha:
            self.parsed[name] = cached
            return cached
        raw = self.executor.source_bytes(name)
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError('Navigation source SHA-256 does not match the recorded version.')
        entry = {'symbols': [], 'imports': [], 'digest': sha,
                 'total_lines': len(raw.decode('utf-8-sig').splitlines())}
        status = 'unsupported_language'
        if name.endswith('.py'):
            status = 'size_limit'
            if len(raw) <= MAX_FILE_BYTES:
                try:
                    entry.update(_parse(raw, name))
                    status = 'parsed'
                except (SyntaxError, UnicodeError, ValueError, RecursionError) as exc:
                    status = type(exc).__name__
        if status != 'parsed':
            entry['parser'] = {
                'status': status, 'complete': False,
                'continuation': {'path': name, 'offset': 0, 'limit': 200},
            }
            if status == 'size_limit':
                entry['parser'].update(file_bytes=len(raw), max_bytes=MAX_FILE_BYTES)
        self.parsed[name] = entry
        return entry

    def _current(self, touched, receipts):
        """只选择本次实际读取或命中的范围，不借后续查询补发其他条目。"""
        updates = []
        for name, ranges in touched.items():
            item = self.state[name]
            spans = merged([*ranges['read'], *ranges['locate']])
            seen = receipts.get(name, _receipt())
            ids = set(seen['ids'])
            update = {'path': name, 'sha256': item['sha256']}
            for field in ('symbols', 'dependencies'):
                nodes = [node for node in item[field] if node['id'] not in ids and any(
                    start < node['end_line'] and end >= node['line'] for start, end in spans)]
                if nodes:
                    update['new_' + field] = nodes
            if not item['parser']['complete'] and not seen['parser']:
                update['parser'] = item['parser']
            if len(update) > 2:
                updates.append(update)
        return updates

    def coverage(self):
        """完整统计留在本地；模型消息仅附带上次确认后发生变化的计数。"""
        total = len(self.executor.inventory())
        full = sum(item['read_ranges'] == [[0, item['total_lines']]]
                   or item['total_lines'] == 0 for item in self.state.values())
        return {
            'total_files': total, 'explored_files': len(self.state), 'fully_read_files': full,
            'status': 'complete' if full == total else 'partial',
            'parsed_files': sum(item['parser']['complete'] for item in self.state.values()),
            'discovered_symbols': sum(len(item['symbols']) for item in self.state.values()),
            'delivered_symbols': sum(
                len({node['id'] for node in item['symbols']} & set(item['delivered']['ids']))
                for item in self.state.values()),
        }

    def acknowledge(self, mapping):
        """只确认实际内联到工具历史的条目；卸载引用和预览不算正文投递。"""
        if not isinstance(mapping, dict) or 'updates' not in mapping:
            return
        receipts = {name: item['delivered'] for name, item in self.state.items()}
        _accept(receipts, mapping['updates'], self.state)
        if mapping.get('source_version') == self.version:
            self.reported.update(mapping.get('coverage_changes', {}))

    def observe(self, data, *, queued=(), record=None):
        files, touched = self.sync(), {}
        for page in read_pages(data):
            if page['path'] in files and files[page['path']] == page['sha256']:
                touched.setdefault(page['path'], {'read': [], 'locate': []})['read'].extend(
                    page['visible_ranges']
                )
        for hit in data.get('hits', []):
            if (hit['path'] in files
                    and hit.get('source_sha256') == files[hit['path']]):
                touched.setdefault(hit['path'], {'read': [], 'locate': []})['locate'].append(
                    [hit['line'] - 1, hit['line']]
                )
        for name, ranges in touched.items():
            entry = self.entry(name, files[name])
            item = self.state.setdefault(name, {
                'sha256': files[name], 'read_ranges': [], 'located_ranges': [],
                'symbols': [], 'dependencies': [], 'delivered': _receipt(),
            })
            item.update(parser=entry['parser'], total_lines=entry['total_lines'])
            for kind, field in (('read', 'read_ranges'), ('locate', 'located_ranges')):
                item[field] = merged([*item[field], *ranges[kind]])
            observed = merged([*item['read_ranges'], *item['located_ranges']])
            item['symbols'] = [s for s in entry['symbols'] if any(
                start < s['end_line'] and end >= s['line'] for start, end in observed
            )]
            dependencies = []
            for value in entry['imports']:
                if not any(start < value['end_line'] and end >= value['line']
                           for start, end in observed):
                    continue
                target = value['target']
                while target and target not in self.modules:
                    target = target.rpartition('.')[0]
                if target and self.modules[target] != name:
                    dependencies.append({
                        'id': value['id'], 'path': self.modules[target],
                        'line': value['line'], 'end_line': value['end_line'],
                        'import_target': value['target'],
                    })
            item['dependencies'] = dependencies

        receipts = {name: deepcopy(item['delivered']) for name, item in self.state.items()}
        reported = dict(self.reported)
        for mapping in queued:
            if isinstance(mapping, dict):
                _accept(receipts, mapping.get('updates', []), self.state)
                if mapping.get('source_version') == self.version:
                    reported.update(mapping.get('coverage_changes', {}))
        pending = self._current(touched, receipts)
        if not pending:
            return None
        changes = {key: value for key, value in self.coverage().items()
                   if reported.get(key) != value}

        def envelope(updates):
            shown = {item['path']: item for item in updates}
            remaining = {}
            for item in pending:
                sent = shown.get(item['path'], {})
                rest = {field: len(item.get(field, [])) - len(sent.get(field, []))
                        for field in _FIELDS if len(item.get(field, [])) > len(sent.get(field, []))}
                if 'parser' in item and 'parser' not in sent:
                    rest['parser'] = True
                if rest:
                    remaining[item['path']] = rest
            result = {'source_version': self.version, 'updates': updates,
                      'truncated': bool(remaining)}
            if changes:
                result['coverage_changes'] = changes
            if record:
                result['record'] = record
            if remaining:
                result['remaining'] = remaining
            if result['truncated'] and record:
                result['continuation'] = {'path': record, 'offset': 0, 'limit': 200}
            return result

        updates = []
        for item in pending[:6]:
            update = dict(item)
            for field, cap in _FIELDS.items():
                if field in item:
                    update[field] = item[field][:cap]
            updates.append(update)
            while len(json.dumps(
                envelope(updates), ensure_ascii=False, indent=2,
            ).encode('utf-8')) > MAX_MAP_BYTES:
                field = next((key for key in reversed(_FIELDS) if update.get(key)), None)
                if field is None:
                    updates.pop()
                    break
                update[field].pop()
                if not update[field]:
                    del update[field]
            if not any(update.get(field) for field in _FIELDS) and 'parser' not in update:
                if updates and updates[-1] is update:
                    updates.pop()
        if not updates:
            # 整条导航放不下时保留归档续读，不截短标签或确认未发送节点。
            if record:
                return {
                    'source_version': self.version, 'record': record, 'truncated': True,
                    'reason': 'Navigation entries exceed inline byte budget; read the record.',
                    'continuation': {'path': record, 'offset': 0, 'limit': 200},
                }
            raise ValueError('Repository map exceeds inline budget and has no record.')
        return envelope(updates)
