"""Grow a source map from real lookup results, without another model or live scan."""
import json
from agent.source.index import _parse, _module, MAX_FILE_BYTES
from agent.source.paging import merged, read_pages


class ExplorationMap:
    def __init__(self, executor, state):
        self.executor, self.state = executor, state
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
        navigator = self.executor.navigator
        cached = navigator.entries.get(name) if navigator and navigator.version == self.version else None
        if cached and cached['digest'] == sha:
            self.parsed[name] = cached
            return cached
        raw = self.executor.source_bytes(name)
        entry = {'symbols': [], 'imports': [], 'calls': [], 'digest': sha, 'error': None}
        if name.endswith('.py') and len(raw) <= MAX_FILE_BYTES:
            try:
                entry.update(_parse(raw, name))
            except (SyntaxError, UnicodeError, ValueError, RecursionError) as exc:
                entry['error'] = type(exc).__name__
        elif name.endswith('.py'):
            entry['error'] = 'file exceeds bounded parser size; text lookup remains available'
        self.parsed[name] = entry
        return entry

    def observe(self, data):
        files, touched = self.sync(), {}
        for page in read_pages(data):
            if page['path'] in files and files[page['path']] == page['sha256']:
                touched.setdefault(page['path'], {'read': [], 'locate': []})['read'].extend(page['visible_ranges'])
        for hit in data.get('hits', []):
            if hit['path'] in files:
                touched.setdefault(hit['path'], {'read': [], 'locate': []})['locate'].append([hit['line'] - 1, hit['line']])
        changes = []
        for name, ranges in touched.items():
            sha = files[name]
            item = self.state.setdefault(name, {'sha256': sha, 'read_ranges': [], 'located_ranges': [],
                                                'symbols': [], 'dependencies': []})
            old_symbols, old_dependencies = list(item['symbols']), list(item['dependencies'])
            item['read_ranges'] = merged([*item['read_ranges'], *ranges['read']])
            item['located_ranges'] = merged([*item['located_ranges'], *ranges['locate']])
            observed = merged([*item['read_ranges'], *item['located_ranges']])
            entry = self.entry(name, sha)
            item['parse_error'] = entry['error']
            item['symbols'] = [s for s in entry['symbols'] if any(
                start < s['end_line'] and end >= s['line'] for start, end in observed)]
            imports = [value for value in entry['imports'] if any(
                start < value['line'] <= end for start, end in observed)]
            for value in imports:
                target = value['target']
                while target and target not in self.modules:
                    target = target.rpartition('.')[0]
                if target and self.modules[target] != name:
                    relation = {'path': self.modules[target], 'line': value['line'],
                                'evidence': 'import ' + value['target'], 'certainty': 'static_candidate'}
                    if relation not in item['dependencies']:
                        item['dependencies'].append(relation)
            changes.append({'path': name, 'read_ranges': ranges['read'], 'located_ranges': ranges['locate'],
                'new_symbols': [{key: s[key] for key in ('name', 'kind', 'signature', 'line', 'end_line')}
                                for s in item['symbols'] if s not in old_symbols][:5],
                'new_dependencies': [r for r in item['dependencies'] if r not in old_dependencies][:3]})
        result = {'source_version': self.version, 'known_files': len(self.state),
                'known_symbols': sum(len(value['symbols']) for value in self.state.values()),
                'updates': changes[:6], 'updates_omitted': len(changes) > 6,
                'note': 'Map grows from actual lookups; static structure is not proof of behavior or test coverage.'}
        while len(json.dumps(result, ensure_ascii=False)) > 4000 and result['updates']:
            result['updates'].pop()
            result['updates_omitted'] = True
        return result
