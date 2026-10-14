"""有界识别本程序记录与分发归档；源码和安装资料共用这一排除边界。"""

import json
from pathlib import Path
import re
import stat
import struct
import zipfile


METADATA_LIMIT = 64 * 1024
DIRECTORY_LIMIT = 2 * 1024 * 1024
ARCHIVE_NAMES = re.compile(
    r'(?:(?:before|after|evaluated)-source|MonoPlanner-\d+\.\d+\.\d+-'
    r'(?:Windows11-x64|Source|source))\.zip', re.IGNORECASE)


class ArtifactScopeError(ValueError):
    pass


def _regular(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(info.st_mode) and not bool(
        getattr(info, 'st_file_attributes', 0)
        & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))


def _json(path, limit=METADATA_LIMIT):
    if not _regular(path) or path.stat().st_size > limit:
        return {}
    try:
        with path.open('rb') as stream:
            value = json.loads(stream.read(limit + 1))
        return value if isinstance(value, dict) else {}
    except (ValueError, UnicodeError):
        return {}


def _portable(value):
    return (value.get('product') == 'MonoPlanner'
            and isinstance(value.get('version'), str)
            and isinstance(value.get('source_fingerprint'), str)
            and re.fullmatch(r'[0-9a-f]{64}', value['source_fingerprint']))


def generated_records(directory):
    """只检查固定的小元数据；不递归扫描、解压或哈希记录原文。"""
    directory = Path(directory)
    state = _json(directory / 'state.json')
    if (state.get('format') == 'task-history-reference'
            and state.get('database') == 'history.sqlite3'
            and _regular(directory / 'history.sqlite3')):
        return True
    manifest_path = directory / 'manifest.json'
    if _regular(manifest_path) and manifest_path.stat().st_size > DIRECTORY_LIMIT:
        with manifest_path.open('rb') as stream:
            prefix = stream.read(METADATA_LIMIT)
            stream.seek(max(0, manifest_path.stat().st_size - METADATA_LIMIT))
            suffix = stream.read(METADATA_LIMIT)
        if ((b'"source_root"' in prefix and b'"status"' in prefix)
                or b'SourceVersions archived bytes; no live workspace read' in suffix):
            raise ArtifactScopeError('源码归档清单超过安全识别上限，请选择实际项目目录；未复制归档。')
    manifest = _json(manifest_path, DIRECTORY_LIMIT)
    if (manifest.get('origin') == 'SourceVersions archived bytes; no live workspace read'
            and re.fullmatch(r'[0-9a-f]{64}', str(manifest.get('source_version', '')))
            and manifest.get('status') in {'capturing', 'captured', 'failed'}
            and isinstance(manifest.get('files'), dict)
            and isinstance(manifest.get('excluded_files'), dict)
            and (directory / 'source').is_dir()):
        return True
    if (manifest.get('status') in {'capturing', 'captured', 'failed'}
            and isinstance(manifest.get('source_root'), str)
            and isinstance(manifest.get('root'), str)
            and isinstance(manifest.get('files'), dict)
            and isinstance(manifest.get('excluded_files'), dict)
            and isinstance(manifest.get('excluded'), list)
            and isinstance(manifest.get('notes'), list)
            and (directory / 'source').is_dir()):
        return True
    if _portable(_json(directory / 'portable.json')) and _regular(directory / 'MonoPlanner.exe'):
        return True
    # 历史评测归档没有产品元数据，以源码包中的产品标识和原哈希表联合确认。
    hashes = _json(directory / 'source-sha256.json', DIRECTORY_LIMIT)
    if hashes and all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)
                      for value in hashes.values()):
        archive = directory / 'evaluated-source.zip'
        if _regular(archive) and generated_archive(archive):
            return True
    baseline = _json(directory / 'baseline.json')
    if re.fullmatch(r'[0-9a-f]{64}', str(baseline.get('code_key', ''))):
        archive = directory / 'before-source.zip'
        if _regular(archive) and generated_archive(archive):
            return True
    # 当前应用数据也可能被用户移动。仅凭同名文件不够，要求 SQLite 的产品表标识。
    database = directory / 'application.sqlite3'
    if _regular(database):
        with database.open('rb') as stream:
            header = stream.read(METADATA_LIMIT)
        if header.startswith(b'SQLite format 3\0') and all(
            marker in header for marker in (b'app_conversations', b'app_messages', b'app_commands')
        ):
            return True
    return False


def generated_archive(path):
    """仅检查产品产物名；陌生或过大的目录结构停止复制，保留用户原件。"""
    path = Path(path)
    if not ARCHIVE_NAMES.fullmatch(path.name):
        return False
    if not _regular(path):
        raise ArtifactScopeError(f'归档不是普通文件，未开始复制：{path.name}')
    try:
        # ZipFile 构造会整体读取中央目录，先核验目录大小，避免识别本身膨胀内存。
        with path.open('rb') as stream:
            size = path.stat().st_size
            stream.seek(max(0, size - 65557))
            tail = stream.read(65557)
            offset = tail.rfind(b'PK\x05\x06')
            if offset < 0 or len(tail) - offset < 22:
                raise ValueError('missing zip directory')
            end = struct.unpack('<4s4H2LH', tail[offset:offset + 22])
            if (end[1] or end[2] or end[3] != end[4] or end[4] > 20000
                    or end[5] > DIRECTORY_LIMIT or end[6] + end[5] > size
                    or offset + 22 + end[7] != len(tail)):
                raise ValueError('archive directory exceeds inspection boundary')
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())

            def read(name):
                info = archive.getinfo(name)
                if info.file_size > METADATA_LIMIT or info.compress_size > METADATA_LIMIT:
                    raise ValueError('archive metadata exceeds inspection boundary')
                with archive.open(info) as stream:
                    return stream.read(METADATA_LIMIT + 1)

            for name in names:
                if name == 'portable.json' or name.endswith('/portable.json'):
                    prefix = name[:-len('portable.json')]
                    if prefix + 'MonoPlanner.exe' in names and _portable(json.loads(read(name))):
                        return True
            required = {'agent/task/runner.py', 'agent/storage/history.py',
                        'resources/prompts/task.txt', 'agent/version.py', 'web/package.json'}
            if required <= names:
                package = json.loads(read('web/package.json'))
                if (package.get('name') == 'monoplanner-web'
                        and b'task-history-reference' in read('agent/storage/history.py')):
                    return True
    except (ValueError, KeyError, UnicodeError, zipfile.BadZipFile, NotImplementedError):
        pass
    raise ArtifactScopeError(
        f'发现未能确认来源的程序同名归档：{path.name}。请核对并选择实际项目目录；'
        '原件未删除，未将该归档复制为源码。')
