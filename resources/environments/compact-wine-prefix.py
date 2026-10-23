"""镜像构建时将与 Wine 原始 PE 文件逐字相同的副本改为只读镜像引用。"""
import hashlib
from pathlib import Path

prefix = Path('/opt/monoplanner/wine-template')
# Wine 默认把文档/桌面映射到 Linux HOME；一次性兼容环境必须使用自己的空目录。
for path in (prefix / 'drive_c/users').rglob('*'):
    if path.is_symlink() and not path.resolve().is_relative_to(prefix / 'drive_c'):
        path.unlink()
        path.mkdir()
libraries = Path('/usr/lib/x86_64-linux-gnu/wine/x86_64-windows')
candidates = {p.name.lower(): p for p in libraries.iterdir() if p.is_file()}
linked = 0
for path in (prefix / 'drive_c').rglob('*'):
    if not path.is_file() or path.is_symlink():
        continue
    source = candidates.get(path.name.lower())
    if (source is None or source.stat().st_size != path.stat().st_size
            or hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(path.read_bytes()).digest()):
        continue
    path.unlink()
    path.symlink_to(source)
    linked += 1
size = sum(p.stat().st_size for p in prefix.rglob('*') if p.is_file() and not p.is_symlink())
if not linked or size > 32 * 1024**2:
    raise RuntimeError('Wine prefix exceeds the bounded temporary environment')
print({'linked_original_libraries': linked, 'writable_prefix_bytes': size})
