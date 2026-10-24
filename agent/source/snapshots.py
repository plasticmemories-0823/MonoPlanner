"""为准备和验收提供独立源码视图，并保留明确证据。

快照将文件及清单保存在磁盘上，不构成安全沙盒；不可信命令应使用 Docker 后端。
仅在源码工作线程暂停时捕获快照；捕获期间检测到并发文件修改就失败。"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import uuid

from agent.execution.docker import DockerEnvironment
from agent.execution.local import LocalEnvironment
from agent.source.artifacts import generated_records, generated_archive
from agent.source.errors import source_storage_boundary
from agent.source.scope import ProjectScope, ProjectScopeError, project_scope as resolve_scope


_DERIVED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".vs",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "htmlcov",
    ".coding_agent",
    "build",
    "dist",
}
_DERIVED_PATHS = {("eval", "runs"), ("eval", "checks"), ("eval", "vendor")}
_SECRET_REASONS = {"environment_credentials", "possible_private_key", "private_key_content"}
_PRIVATE_MARKERS = (
    b"-----BEGIN PRIVATE KEY-----",
    b"-----BEGIN RSA PRIVATE KEY-----",
    b"-----BEGIN EC PRIVATE KEY-----",
    b"-----BEGIN OPENSSH PRIVATE KEY-----",
    b"-----BEGIN DSA PRIVATE KEY-----",
    b"-----BEGIN ENCRYPTED PRIVATE KEY-----",
)


class SnapshotError(RuntimeError):
    """无法忠实复制源码，或已保存的版本发生变化。"""


# ========================================================================
# 路径、链接与文件排除规则
# ========================================================================
def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _link(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _reason(relative):
    parts = tuple(part.lower() for part in relative.parts)
    name = parts[-1]
    # 敏感文件排除规则优先于生成目录规则；受版本控制的普通文件
    # 可以覆盖目录规则，但不能覆盖此排除规则。
    if name == ".env" or name.startswith(".env."):
        return "environment_credentials"
    if name in {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"} or relative.suffix.lower() in {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".dpapi",
    }:
        return "possible_private_key"
    if any(part in _DERIVED_DIRS for part in parts[:-1]):
        return "generated_directory"
    if parts[:2] in _DERIVED_PATHS:
        return "run_artifacts_or_downloaded_dependencies"
    if name == ".coverage" or relative.suffix.lower() in {".pyc", ".pyo", ".log"}:
        return "generated_file"
    return None


def _inspect_file(path, cancel=None):
    from agent.storage.resources import owner_cancelled

    digest, previous = hashlib.sha256(), b""
    private = False
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if owner_cancelled() or (cancel is not None and cancel.is_set()):
                raise InterruptedError('源码哈希检查已中断。')
            digest.update(chunk)
            data = previous + chunk
            private = private or any(line in _PRIVATE_MARKERS for line in data.splitlines())
            previous = data[-80:]
    return digest.hexdigest(), private


def _safe_file(root, relative):
    if relative.is_absolute() or ".." in relative.parts:
        raise SnapshotError(f"Unsafe source path: {relative}")
    path = root
    for part in relative.parts:
        path /= part
        if _link(path):
            raise SnapshotError(f"Links and Windows reparse points are not supported: {relative}")
    if not path.is_file():
        raise SnapshotError(f"Source is not a regular file: {relative}")
    if not _within(path.resolve(), root):
        raise SnapshotError(f"Source escapes repository: {relative}")
    return path


# ========================================================================
# 源码发现与版本控制文件选择
# ========================================================================
def _generated_records(directory):
    return generated_records(directory)


def _walk(root, preserve_files=(), ignored_directories=(), cancel=None,
          project_scope=None, apply_ignores=False):
    """即使链接指向内部也不跟随；报告链接而非复制其目标。"""
    retained_directories = {parent for name in preserve_files for parent in Path(name).parents}
    ignored_directories = set(ignored_directories)
    scope = project_scope or resolve_scope(root, cancel=cancel)
    rules = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        from agent.storage.resources import owner_cancelled

        if owner_cancelled() or (cancel is not None and cancel.is_set()):
            raise InterruptedError('源码扫描已中断。')
        parent = Path(directory)
        if (parent != root and not scope.excludes(parent.relative_to(root))
                and ((parent / '.git').exists() or (parent / '.git').is_symlink())):
            raise ProjectScopeError('调查期间出现新的独立项目，已停止；请核对后新建任务。')
        inherited = list(rules.get(parent.parent, []))
        if apply_ignores and (parent / '.gitignore').is_file():
            from pathspec import GitIgnoreSpec

            ignore = _safe_file(root, (parent / '.gitignore').relative_to(root))
            if ignore.stat().st_size > 1024 * 1024:
                raise SnapshotError('项目忽略文件过大，无法确定源码范围。')
            inherited.append((parent, GitIgnoreSpec.from_lines(
                ignore.read_text('utf-8-sig').splitlines())))
        rules[parent] = inherited

        def ignored(path, directory=False):
            decision = None
            for base, spec in inherited:
                match = spec.check_file(path.relative_to(base).as_posix()
                                        + ('/' if directory else ''))
                if match.include is not None:
                    decision = match.include
            return decision is True
        for name in list(directories):
            relative = (parent / name).relative_to(root)
            if scope.excludes(relative):
                directories.remove(name)
                yield relative, "independent_project"
            elif apply_ignores and relative not in retained_directories and ignored(parent / name, True):
                directories.remove(name)
                yield relative, "git_ignored_directory"
            elif relative in ignored_directories and relative not in retained_directories:
                directories.remove(name)
                yield relative, "git_ignored_directory"
            elif relative not in retained_directories and (
                name.lower() in _DERIVED_DIRS
                or tuple(part.lower() for part in relative.parts[:2]) in _DERIVED_PATHS
            ):
                directories.remove(name)
                yield relative, "generated_directory"
            elif _link(parent / name):
                raise SnapshotError(
                    f"Links and Windows reparse points are not supported: {relative}"
                )
            elif relative not in retained_directories and _generated_records(parent / name):
                directories.remove(name)
                yield relative, 'monoplanner_generated_records'
        for name in files:
            if owner_cancelled() or (cancel is not None and cancel.is_set()):
                raise InterruptedError('源码扫描已中断。')
            relative = (parent / name).relative_to(root)
            if scope.excludes(relative):
                continue
            if apply_ignores and relative not in preserve_files and ignored(parent / name):
                if _reason(relative) not in _SECRET_REASONS:
                    yield relative, "git_ignored_file"
                    continue
            skip = ('monoplanner_generated_archive'
                    if relative not in preserve_files and not _reason(relative)
                    and generated_archive(_safe_file(root, relative)) else None)
            yield relative, skip


def _selected_files(root, runtime_files=(), cancel=None, project_scope=None):
    """通过 Git 纳入暂存、已修改和新增文件；后备方案按排除规则遍历源码。"""
    scope = resolve_scope(root, project_scope, cancel=cancel)
    for name in runtime_files:
        try:
            scope.require(name)
        except ProjectScopeError as exc:
            raise SnapshotError(str(exc)) from exc
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    # 源码发现属于可信主机操作；不能仅因 Git 读取索引，
    # 就在主机上执行仓库的 fsmonitor 钩子。
    git = ["git", "-c", "core.fsmonitor=false", "-C", str(root)]
    try:
        probe = subprocess.run(
            [*git, "rev-parse", "--show-toplevel"], capture_output=True, timeout=10, **options
        )
    except FileNotFoundError:
        probe = None
    git_root = (Path(os.fsdecode(probe.stdout.strip())).resolve()
                if probe is not None and probe.returncode == 0 else None)
    use_git = git_root is not None and root.is_relative_to(git_root)
    if use_git and git_root != root:
        ignored = subprocess.run([*git, "check-ignore", "-q", "--", "."],
                                 capture_output=True, timeout=10, **options)
        if ignored.returncode not in {0, 1}:
            raise SnapshotError("git check-ignore failed for selected project directory.")
        # 用户明确选中的独立目录可能被外层仓库整体忽略；仍保留其自身源码。
        use_git = ignored.returncode == 1
    if use_git:
        listed = subprocess.run(
            [
                *git,
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                ".",
            ],
            capture_output=True,
            timeout=30,
            **options,
        )
        if listed.returncode != 0:
            raise SnapshotError("git ls-files failed; source selection was not complete.")
        tracked_result = subprocess.run(
            [*git, "ls-files", "--cached", "-z", "--", "."], capture_output=True, timeout=30, **options
        )
        if tracked_result.returncode != 0:
            raise SnapshotError("git ls-files failed to distinguish tracked source files.")
        tracked = {Path(os.fsdecode(item)) for item in tracked_result.stdout.split(b"\0") if item}
        selected = {Path(os.fsdecode(item)) for item in listed.stdout.split(b"\0") if item}
        selected = {path for path in selected if not scope.excludes(path)}
        tracked = {path for path in tracked if not scope.excludes(path)}
        ignored_result = subprocess.run(
            [
                *git,
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--directory",
                "-z",
                "--",
                ".",
            ],
            capture_output=True,
            timeout=30,
            **options,
        )
        if ignored_result.returncode != 0:
            raise SnapshotError("git ls-files failed to identify ignored directories.")
        ignored_directories = {
            Path(os.fsdecode(item).rstrip("/"))
            for item in ignored_result.stdout.split(b"\0")
            if item.endswith(b"/")
        }
        for name in runtime_files:
            relative = Path(name)
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or _reason(relative)
            ):
                raise SnapshotError(f"Invalid runtime material path: {name}")
            selected.add(relative)
        # 即使 Git 忽略指定的敏感文件，也要在清单记录其排除项；
        # 通过哈希，使后续候选能够安全报告未验证的改动。
        selected.update(
            relative
            for relative, skip in _walk(root, tracked, ignored_directories, cancel=cancel,
                                       project_scope=scope)
            if not skip and _reason(relative) in _SECRET_REASONS
        )
        retained, generated, checked = [], set(), {}
        for relative in sorted(selected):
            record_root = None
            if relative not in tracked:
                for parent in relative.parents:
                    if parent == Path('.'):
                        continue
                    if parent not in checked:
                        checked[parent] = _generated_records(root / parent)
                    if checked[parent]:
                        record_root = parent
                        break
            if record_root is None:
                skip = ('monoplanner_generated_archive'
                        if relative not in tracked and not _reason(relative)
                        and generated_archive(_safe_file(root, relative)) else None)
                retained.append((relative, skip, relative in tracked))
            else:
                generated.add(record_root)
        retained.extend((Path(name), "independent_project", False)
                        for name in scope.excluded_projects)
        return "git", retained + [
            (relative, 'monoplanner_generated_records', False) for relative in sorted(generated)
        ]
    if _generated_records(root):
        raise SnapshotError('所选目录是 MonoPlanner 运行记录，请选择实际工作项目。')
    return "source_walk", [
        (relative, skip, False)
        for relative, skip in _walk(root, cancel=cancel, project_scope=scope, apply_ignores=True)
    ]


def _save(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(payload, ensure_ascii=True, indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# ========================================================================
# 快照捕获、隔离副本与完整性校验
# ========================================================================
@dataclass(frozen=True)
class SourceSnapshot:
    root: Path
    files: dict[str, str]
    digest: str
    manifest_path: Path
    excluded_files: dict[str, str] = field(default_factory=dict)

    @classmethod
    @source_storage_boundary
    def capture(cls, repo_root, destination, excluded_roots=(), runtime_files=(), cancel=None,
                project_scope=None):
        source = Path(repo_root).expanduser().resolve()
        destination = Path(destination).expanduser().resolve()
        if not source.is_dir():
            raise SnapshotError(f"Source repository is not a directory: {source}")
        if _within(destination, source):
            raise SnapshotError("Snapshot destination must be outside the source repository.")
        if destination.exists():
            raise SnapshotError("Snapshot destination already exists; refusing to overwrite it.")
        excluded = [Path(item).expanduser().resolve() for item in excluded_roots]
        if any(_within(source, item) for item in excluded):
            raise SnapshotError("An excluded root would exclude the entire source repository.")
        destination.mkdir(parents=True, exist_ok=False)
        root = destination / "source"
        root.mkdir()
        manifest_path = destination / "manifest.json"
        manifest = {
            "source_root": str(source),
            "root": str(root),
            "status": "capturing",
            "files": {},
            "excluded_files": {},
            "excluded": [],
            "notes": [
                "Git selection stays inside the selected directory; independently ignored roots use their own ignore rules.",
                "Tracked ordinary source files are retained even under names usually used for generated files.",
                "Git-ignored untracked files are omitted; named secret files outside generated directories are fingerprinted.",
                "Environment secrets, private keys, links, VCS data and generated caches are not copied.",
                "Run capture while the worker is paused; this is not a transactional filesystem snapshot.",
            ],
        }
        _save(manifest_path, manifest)
        try:
            from agent.storage.resources import owner_cancelled, preflight_space

            def check_cancel():
                if owner_cancelled() or (cancel is not None and cancel.is_set()):
                    raise InterruptedError('源码快照已中断；部分现场已保留。')

            check_cancel()
            scope = resolve_scope(source, project_scope, cancel=cancel)
            manifest["project_scope"] = scope.to_record()
            selection, candidates = _selected_files(
                source, runtime_files, cancel=cancel, project_scope=scope)
            manifest["selection"] = selection
            manifest["runtime_files"] = sorted(runtime_files)
            manifest["tracked_files"] = sorted(
                relative.as_posix() for relative, _, tracked in candidates if tracked
            )
            required = 0
            for relative, skip, tracked in candidates:
                check_cancel()
                path = source / relative
                reason = skip or _reason(relative)
                if tracked and reason not in _SECRET_REASONS:
                    reason = None
                if reason or any(_within(path.resolve(), item) for item in excluded):
                    continue
                if path.exists():
                    required += _safe_file(source, relative).stat().st_size
            manifest['space_budget'] = preflight_space(destination, required)
            for relative, skip, tracked in candidates:
                check_cancel()
                path = source / relative
                skip = skip or _reason(relative)
                if tracked and skip not in _SECRET_REASONS:
                    skip = None
                if any(_within(path.resolve(), item) for item in excluded):
                    skip = "explicit_artifact_root"
                if not path.exists() and not path.is_symlink():
                    skip = "deleted_tracked_file"
                if skip:
                    manifest["excluded"].append({"path": relative.as_posix(), "reason": skip})
                    if path.is_file() and not skip.startswith(('monoplanner_generated_', 'git_ignored_')):
                        safe = _safe_file(source, relative)
                        manifest["excluded_files"][relative.as_posix()] = _inspect_file(safe, cancel)[0]
                    continue
                path = _safe_file(source, relative)
                before = path.stat()
                sha, private = _inspect_file(path, cancel)
                if private:
                    manifest["excluded"].append(
                        {"path": relative.as_posix(), "reason": "private_key_content"}
                    )
                    manifest["excluded_files"][relative.as_posix()] = sha
                    continue
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with path.open('rb') as original, target.open('xb') as output:
                    while chunk := original.read(1024 * 1024):
                        check_cancel()
                        output.write(chunk)
                shutil.copystat(path, target)
                copied_sha, copied_private = _inspect_file(target, cancel)
                after = path.stat()
                if copied_private:
                    # 此文件是在快照内部新建的，不能删除源码数据。
                    target.resolve().relative_to(root)
                    target.unlink()
                    raise SnapshotError(
                        f"Source changed to secret content during capture: {relative}"
                    )
                if (
                    copied_sha != sha
                    or before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns
                ):
                    raise SnapshotError(f"Source changed during capture: {relative}")
                manifest["files"][relative.as_posix()] = sha
            digest = hashlib.sha256(
                json.dumps(
                    {"files": manifest["files"], "excluded_files": manifest["excluded_files"]},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            manifest.update(status="captured", digest=digest, file_count=len(manifest["files"]))
            _save(manifest_path, manifest)
            return cls(
                root,
                dict(manifest["files"]),
                digest,
                manifest_path,
                dict(manifest["excluded_files"]),
            )
        except Exception as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            try:
                _save(manifest_path, manifest)
            except OSError:
                # 写失败说明也失败时，保留上一份原子清单和最初异常，不能截断原件。
                pass
            raise

    @source_storage_boundary
    def build_copy(self, destination):
        """按原清单复制编译工作区；不能重新扫描而丢失已跟踪的输入。"""
        from agent.storage.resources import owner_cancelled, preflight_space

        destination = Path(destination).expanduser().resolve()
        if _within(destination, self.root):
            raise SnapshotError("Build workspace must be outside the evidence snapshot.")
        if destination.exists():
            raise SnapshotError("Build workspace already exists; refusing to overwrite it.")
        if owner_cancelled():
            raise InterruptedError("Build workspace copy interrupted; evidence is retained.")
        if not self.verify_integrity()["valid"]:
            raise SnapshotError("Source evidence changed before creating the build workspace.")
        manifest = json.loads(self.manifest_path.read_bytes())
        if manifest["files"] != self.files or manifest["digest"] != self.digest:
            raise SnapshotError("Source manifest differs from the captured evidence.")
        required = sum(_safe_file(self.root, Path(name)).stat().st_size for name in self.files)
        destination.mkdir(parents=True, exist_ok=False)
        root = destination / "source"
        root.mkdir()
        manifest.update(root=str(root), source_root=str(self.root), status="copying",
                        evidence_manifest=str(self.manifest_path),
                        space_budget=preflight_space(destination, required))
        path = destination / "manifest.json"
        _save(path, manifest)
        for name, expected in self.files.items():
            source = _safe_file(self.root, Path(name))
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as original, target.open("xb") as output:
                while chunk := original.read(1024 * 1024):
                    if owner_cancelled():
                        raise InterruptedError("Build workspace copy interrupted; evidence is retained.")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            shutil.copystat(source, target)
            sha, private = _inspect_file(target)
            if sha != expected or private:
                raise SnapshotError(f"Source evidence changed while copying: {name}")
        if not self.verify_integrity()["valid"]:
            raise SnapshotError("Source evidence changed while creating the build workspace.")
        manifest["status"] = "captured"
        _save(path, manifest)
        return type(self)(root, dict(self.files), self.digest, path, dict(self.excluded_files))

    def verify_build_integrity(self, output_paths=()):
        """仅编译副本允许已知派生物变化；原件和显式输入仍按字节保护。"""
        report = self.verify_integrity()
        manifest = json.loads(self.manifest_path.read_bytes())
        protected = set(manifest.get("runtime_files", ()))
        # 旧 Git 清单没有区分跟踪文件，保守保留全部输入保护。
        protected.update(manifest.get("tracked_files", self.files if manifest.get("selection") == "git" else ()))
        coq_sources = {Path(name) for name in self.files if Path(name).suffix == ".v"}
        outputs = set(output_paths)

        def derived(name):
            if name in protected:
                return False
            path = Path(name)
            cache = any(part in {'.vitest-cache', '.vite-cache'} for part in path.parts[:-1])
            if (name in outputs or cache
                    or _reason(path) in {'generated_directory', 'generated_file'}):
                return True
            if path.suffix in {".glob", ".vo", ".vos", ".vok"}:
                return path.with_suffix(".v") in coq_sources
            if path.name.startswith(".") and path.name.endswith(".aux"):
                return path.with_name(path.name[1:-4] + ".v") in coq_sources
            return path.name in {".lia.cache", ".nia.cache"} and any(
                source.parent == path.parent for source in coq_sources
            )

        report["build_outputs"] = {}
        for kind in ("added", "deleted", "modified"):
            report["build_outputs"][kind] = [name for name in report[kind] if derived(name)]
            report[kind] = [name for name in report[kind] if not derived(name)]
        report["valid"] = not report["errors"] and not any(
            report[kind] for kind in ("added", "deleted", "modified")
        )
        return report

    @contextmanager
    def environment_like(self, environment, artifact_root, *, read_only=False):
        """保留执行后端；仅管理和关闭新建的容器。"""
        if isinstance(environment, DockerEnvironment):
            child = DockerEnvironment(
                self.root,
                environment.image,
                f"agent-snapshot-{uuid.uuid4().hex}",
                timeout=environment.timeout,
                artifact_root=artifact_root,
                **environment.clone_options(),
            )
            child.repo_read_only = read_only
            child.root_read_only = read_only
            child.disposable = True
            try:
                child.start()
                yield child
            except BaseException:
                child._execution_unknown = True
                raise
            finally:
                child.close()
        elif isinstance(environment, LocalEnvironment):
            child = LocalEnvironment(
                self.root, timeout=environment.timeout, artifact_root=artifact_root
            )
            child.python_executable = environment.python_executable
            yield child
        else:
            raise TypeError(
                "Snapshot supports LocalEnvironment and DockerEnvironment backends only."
            )

    def verify_integrity(self):
        """检查源码文件和新增源码路径，忽略派生缓存。"""
        current, errors = {}, []
        try:
            if not self.root.is_dir() or _link(self.root):
                raise SnapshotError("Source snapshot root is missing or replaced by a link.")
            for relative, skip in _walk(
                self.root, preserve_files=self.files, project_scope=ProjectScope(self.root)
            ):
                reason = _reason(relative)
                if skip or (
                    relative.as_posix() not in self.files
                    and reason
                    and reason not in _SECRET_REASONS
                ):
                    continue
                path = _safe_file(self.root, relative)
                sha, private = _inspect_file(path)
                current[relative.as_posix()] = sha
                if private:
                    errors.append(f"Private key content appeared: {relative}")
        except (OSError, SnapshotError, ProjectScopeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        change = _changes(self.files, current)
        return {
            "valid": not any(change.values()) and not errors,
            **change,
            "errors": errors,
            "digest": self.digest,
        }

    # ========================================================================
    # 变更比较与可交付补丁
    # ========================================================================
    def changes_from(self, base):
        return _changes(base.files, self.files)

    def excluded_changes_from(self, base):
        """未进入可执行副本的改动不能声称已经验证。"""
        return _changes(base.excluded_files, self.excluded_files)

    def write_patch(self, base, path, max_text_bytes=1024 * 1024):
        """写入 UTF-8 文本改动，并在清单中明确列出省略项。

        二进制、非 UTF-8 和大文件保存在快照中，不嵌入补丁。
        文件权限变化和空目录不属于文本补丁。"""
        for snapshot in (base, self):
            if not snapshot.verify_integrity()["valid"]:
                raise SnapshotError(
                    "Source snapshot changed; refusing to export an unverified patch."
                )
        path = Path(path).resolve()
        if _within(path, base.root) or _within(path, self.root):
            raise SnapshotError("Patch evidence must be outside the source snapshots.")
        changes, lines, omitted = self.changes_from(base), [], []
        excluded_changes = self.excluded_changes_from(base)
        for relative in sorted(set().union(*changes.values())):
            before, after = base.root / relative, self.root / relative
            try:
                if any(char in relative for char in "\t\r\n"):
                    raise ValueError("filename requires unsupported patch quoting")
                if any(
                    item.exists() and item.stat().st_size > max_text_bytes
                    for item in (before, after)
                ):
                    raise ValueError("file exceeds textual patch size limit")
                old = before.read_bytes() if relative in base.files else b""
                new = after.read_bytes() if relative in self.files else b""
                if b"\0" in old or b"\0" in new:
                    raise ValueError("binary file")
                old_text, new_text = old.decode("utf-8"), new.decode("utf-8")
                if not old and not new:
                    raise ValueError(
                        "empty file addition/deletion is not represented by unified text diff"
                    )
                diff = difflib.unified_diff(
                    old_text.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile="a/" + relative if relative in base.files else "/dev/null",
                    tofile="b/" + relative if relative in self.files else "/dev/null",
                )
                for line in diff:
                    lines.append(
                        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                    )
            except (UnicodeError, ValueError) as exc:
                omitted.append({"path": relative, "reason": str(exc)})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes("".join(lines).encode("utf-8"))
        report = {
            "base_digest": base.digest,
            "candidate_digest": self.digest,
            "changes": changes,
            "patch": str(path),
            "omitted": omitted,
            "excluded_changes": excluded_changes,
            "complete_text_patch": not omitted and not any(excluded_changes.values()),
            "limitations": ["File modes and empty directories are not represented."],
        }
        _save(path.with_suffix(path.suffix + ".manifest.json"), report)
        return report


def _changes(before, after):
    return {
        "added": sorted(after.keys() - before.keys()),
        "deleted": sorted(before.keys() - after.keys()),
        "modified": sorted(
            path for path in before.keys() & after.keys() if before[path] != after[path]
        ),
    }
