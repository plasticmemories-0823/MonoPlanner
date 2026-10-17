"""按范围读取当前文本，不创建源码快照、不调用模型决策或执行命令。"""

import hashlib
from pathlib import Path

from agent.source.snapshots import _link, _reason, _PRIVATE_MARKERS, _safe_file, SnapshotError
from agent.source.paging import merged, unread, ranges


def authorized_roots(values):
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError('授权资料目录必须是路径列表')
    result = []
    for value in values:
        path = Path(value).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise ValueError('授权资料目录必须是已有绝对路径')
        if any(_link(p) for p in (path, *path.parents)):
            raise ValueError('授权目录不能经过链接或 Windows 重解析点')
        if str(path.resolve()) not in result:
            result.append(str(path.resolve()))
    return result


def authorized_sources(values):
    if not isinstance(values, list):
        raise ValueError('可读资料必须是文件或目录列表')
    result = []
    for item in values:
        if (
            not isinstance(item, dict)
            or set(item) != {'path', 'kind'}
            or item['kind'] not in {'file', 'directory'}
            or not isinstance(item['path'], str)
        ):
            raise ValueError('资料须包含 path 和 kind（file 或 directory）')
        path = Path(item['path']).expanduser()
        from agent.storage.preferences import private_path

        if private_path(path):
            raise ValueError('不能授权助手私有记录目录')
        if not path.is_absolute() or '..' in path.parts or ':' in str(path)[2:]:
            raise ValueError('资料必须使用明确的已有绝对路径')
        if any(_link(p) for p in (path, *path.parents)):
            raise ValueError('资料不能经过链接或 Windows 重解析点')
        if item['kind'] == 'directory':
            if not path.is_dir():
                raise ValueError('资料目录不存在')
        else:
            if not path.is_file() or path.stat().st_nlink != 1:
                raise ValueError('资料必须是独立普通文件')
            if path.suffix.lower() == '.dpapi' or _reason(Path(path.name)) in {
                'environment_credentials',
                'possible_private_key',
            }:
                raise ValueError('不能授权密钥或环境凭据')
            from agent.source.snapshots import _inspect_file

            if _inspect_file(path)[1]:
                raise ValueError('不能授权含私钥的文件')
        entry = {'path': str(path.resolve()), 'kind': item['kind']}
        if entry not in result:
            result.append(entry)
    return result


class AssistantFiles:
    def __init__(self, workspace, read_roots=(), *, read_sources=()):
        self.workspace = Path(workspace).resolve() if workspace else None
        self.roots = [
            Path(v) for v in authorized_roots(([workspace] if workspace else []) + list(read_roots))
        ]
        self.files = []
        for item in authorized_sources(list(read_sources)):
            (self.roots if item['kind'] == 'directory' else self.files).append(Path(item['path']))

    def path(self, value, *, directory=False):
        path = Path(value)
        if not path.is_absolute():
            if self.workspace is None:
                raise ValueError('相对路径需要工作目录；请指定已授权资料中的绝对路径')
            path = self.workspace / path
        if '..' in path.parts:
            raise ValueError('路径不能包含 ..')
        from agent.storage.preferences import private_path

        if private_path(path):
            raise ValueError('不读取助手私有记录')
        if ':' in str(path)[2:]:
            raise ValueError('不读取备用数据流或特殊设备路径')
        if not directory and path in self.files:
            authorized_sources([{'path': str(path), 'kind': 'file'}])
            return path
        for root in sorted(self.roots, key=lambda p: len(p.parts), reverse=True):
            if path.is_relative_to(root):
                relative = path.relative_to(root)
                if relative.parts and (
                    _reason(relative) in {'environment_credentials', 'possible_private_key'}
                    or relative.suffix.lower() == '.dpapi'
                ):
                    raise ValueError('不读取密钥或环境凭据')
                if any(_link(p) for p in (path, *path.parents)):
                    raise ValueError('不跟随链接或 Windows 重解析点')
                if directory:
                    if not path.is_dir():
                        raise ValueError('不是已有目录')
                    return path
                try:
                    target = _safe_file(root, relative)
                    if target.stat().st_nlink != 1:
                        raise ValueError('不读取多重硬链接')
                    return target
                except SnapshotError as exc:
                    raise ValueError(str(exc)) from exc
        raise ValueError('路径不在工作目录或用户授权资料目录内')

    def bytes(self, path):
        if path.stat().st_size > 2_000_000:
            raise ValueError('单个文本文件最多读取 2 MB')
        raw = path.read_bytes()
        if any(marker in raw for marker in _PRIVATE_MARKERS):
            raise ValueError('文件含有私钥，不读取')
        if b'\0' in raw:
            raise ValueError('仅支持文本文件')
        return raw

    def list(self, path, offset, limit):
        ranges(offset, limit)
        base = self.path(path, directory=True)
        entries = []
        for p in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold())):
            if (
                _link(p)
                or p.suffix.lower() == '.dpapi'
                or _reason(Path(p.name)) in {'environment_credentials', 'possible_private_key'}
            ):
                continue
            entries.append({'name': p.name, 'path': str(p), 'directory': p.is_dir()})
        stop = offset + min(limit, 200)
        return {
            'path': str(base),
            'entries': entries[offset:stop],
            'next_offset': stop if stop < len(entries) else None,
        }

    def read(self, path, offset, limit, history, repeat=False, *, cursor=None):
        from agent.source.paging import ReadBudget, read_pages
        import json

        ranges(offset, limit)
        path = self.path(path)
        raw = self.bytes(path)
        lines = raw.decode('utf-8-sig').splitlines()
        sha = hashlib.sha256(raw).hexdigest()
        budget = ReadBudget(24000, unit='unicode_characters')
        if cursor:
            return {
                'path': str(path),
                'already_visible': False,
                **budget.source(raw, offset, limit, cursor=cursor),
            }
        seen = []
        if not repeat:
            # 只有当前请求视图中的实际读取结果算作已读，本地归档和摘要不算。
            for message in history:
                if message.get('type') != 'tool':
                    continue
                try:
                    data = json.loads(message['data']['content'])
                except (ValueError, KeyError, TypeError):
                    continue
                for page in read_pages(data):
                    if page['path'] == str(path) and page['sha256'] == sha:
                        seen.extend(page['visible_ranges'])
        end = min(len(lines), offset + limit)
        missing = unread(offset, end, merged(seen))
        pages = []
        for start, stop in missing:
            if pages and budget.remaining:
                budget.remaining -= 1
            pages.append(budget.source(raw, start, stop - start))
            if pages[-1]['truncated']:
                break
        visible = merged([pair for page in pages for pair in page['visible_ranges']])
        gaps = unread(offset, end, merged([*seen, *visible]))
        result = {
            'path': str(path),
            'sha256': sha,
            'offset': offset,
            'offset_unit': 'lines',
            'total_lines': len(lines),
            'text': '\n'.join(page['text'] for page in pages if page['text']),
            'visible_ranges': visible,
            'already_visible': not missing,
            'truncated': bool(gaps),
            'next_offset': gaps[0][0] if gaps else end,
            'next_cursor': pages[-1].get('next_cursor') if pages else None,
            'budget_unit': budget.unit,
        }
        if pages:
            for key in ('long_line', 'partial_byte_range', 'byte_offset_unit'):
                if key in pages[-1]:
                    result[key] = pages[-1][key]
        return result
