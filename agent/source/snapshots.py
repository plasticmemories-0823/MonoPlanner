"""Independent source views for preparation and acceptance, with explicit evidence.

Snapshots retain files and manifests on disk. They are not a security sandbox: use
the Docker backend for untrusted commands. Capture only while the source worker
is paused; concurrent file changes detected during capture make it fail.
"""

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


_DERIVED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".vs", ".idea",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox", "htmlcov",
    ".coding_agent", "build", "dist",
}
_DERIVED_PATHS = {("eval", "runs"), ("eval", "checks"), ("eval", "vendor")}
_SECRET_REASONS = {"environment_credentials", "possible_private_key", "private_key_content"}
_PRIVATE_MARKERS = (
    b"-----BEGIN PRIVATE KEY-----", b"-----BEGIN RSA PRIVATE KEY-----",
    b"-----BEGIN EC PRIVATE KEY-----", b"-----BEGIN OPENSSH PRIVATE KEY-----",
    b"-----BEGIN DSA PRIVATE KEY-----", b"-----BEGIN ENCRYPTED PRIVATE KEY-----",
)


class SnapshotError(RuntimeError):
    """Source cannot be copied faithfully or its saved version changed."""


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
    # Secrets must win before generated-directory rules: tracked ordinary files
    # may override those directory rules, but must never override this exclusion.
    if name == ".env" or name.startswith(".env."):
        return "environment_credentials"
    if (name in {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
            or relative.suffix.lower() in {".pem", ".key", ".p12", ".pfx", ".dpapi"}):
        return "possible_private_key"
    if any(part in _DERIVED_DIRS for part in parts[:-1]):
        return "generated_directory"
    if parts[:2] in _DERIVED_PATHS:
        return "run_artifacts_or_downloaded_dependencies"
    if name == ".coverage" or relative.suffix.lower() in {".pyc", ".pyo", ".log"}:
        return "generated_file"
    return None


def _inspect_file(path):
    digest, previous = hashlib.sha256(), b""
    private = False
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
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


def _walk(root, preserve_files=(), ignored_directories=()):
    """Never follow even an internal link; report it instead of copying a target."""
    retained_directories = {parent for name in preserve_files for parent in Path(name).parents}
    ignored_directories = set(ignored_directories)
    for directory, directories, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(directories):
            relative = (parent / name).relative_to(root)
            if relative in ignored_directories and relative not in retained_directories:
                directories.remove(name)
                yield relative, "git_ignored_directory"
            elif (relative not in retained_directories and (name.lower() in _DERIVED_DIRS or tuple(
                    part.lower() for part in relative.parts[:2]) in _DERIVED_PATHS)):
                directories.remove(name)
                yield relative, "generated_directory"
            elif _link(parent / name):
                raise SnapshotError(f"Links and Windows reparse points are not supported: {relative}")
        for name in files:
            yield (parent / name).relative_to(root), None


def _selected_files(root, runtime_files=()):
    """Git includes staged, dirty and new files; fallback is a pruned source walk."""
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    # Source discovery is trusted host work, so repository fsmonitor hooks must
    # not execute on the host merely because Git reads an index.
    git = ["git", "-c", "core.fsmonitor=false", "-C", str(root)]
    try:
        probe = subprocess.run([*git, "rev-parse", "--show-toplevel"],
                               capture_output=True, timeout=10, **options)
    except FileNotFoundError:
        probe = None
    if (probe is not None and probe.returncode == 0
            and Path(os.fsdecode(probe.stdout.strip())).resolve() == root):
        listed = subprocess.run([
            *git, "ls-files", "--cached", "--others",
            "--exclude-standard", "-z", "--", ".",
        ], capture_output=True, timeout=30, **options)
        if listed.returncode != 0:
            raise SnapshotError("git ls-files failed; source selection was not complete.")
        tracked_result = subprocess.run([*git, "ls-files", "--cached", "-z"],
                                        capture_output=True, timeout=30, **options)
        if tracked_result.returncode != 0:
            raise SnapshotError("git ls-files failed to distinguish tracked source files.")
        tracked = {Path(os.fsdecode(item)) for item in tracked_result.stdout.split(b"\0") if item}
        selected = {Path(os.fsdecode(item)) for item in listed.stdout.split(b"\0") if item}
        ignored_result = subprocess.run([
            *git, "ls-files", "--others", "--ignored",
            "--exclude-standard", "--directory", "-z",
        ], capture_output=True, timeout=30, **options)
        if ignored_result.returncode != 0:
            raise SnapshotError("git ls-files failed to identify ignored directories.")
        ignored_directories = {Path(os.fsdecode(item).rstrip("/"))
                               for item in ignored_result.stdout.split(b"\0") if item.endswith(b"/")}
        for name in runtime_files:
            relative = Path(name)
            if not relative.parts or relative.is_absolute() or ".." in relative.parts or _reason(relative):
                raise SnapshotError(f"Invalid runtime material path: {name}")
            selected.add(relative)
        # Include named secret exclusions in the manifest even when Git ignores
        # them. Hashes let a later candidate report unverified changes safely.
        selected.update(relative for relative, skip in _walk(root, tracked, ignored_directories)
                        if not skip and _reason(relative) in _SECRET_REASONS)
        return "git", [(relative, None, relative in tracked) for relative in sorted(selected)]
    return "source_walk", [(relative, skip, False) for relative, skip in _walk(root)]


def _save(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class SourceSnapshot:
    root: Path
    files: dict[str, str]
    digest: str
    manifest_path: Path
    excluded_files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls, repo_root, destination, excluded_roots=(), runtime_files=()):
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
        manifest = {"source_root": str(source), "root": str(root), "status": "capturing",
                    "files": {}, "excluded_files": {}, "excluded": [], "notes": [
                        "Git selection applies only to the selected repository root; independent subdirectories use a source walk.",
                        "Tracked ordinary source files are retained even under names usually used for generated files.",
                        "Git-ignored untracked files are omitted; named secret files outside generated directories are fingerprinted.",
                        "Environment secrets, private keys, links, VCS data and generated caches are not copied.",
                        "Run capture while the worker is paused; this is not a transactional filesystem snapshot.",
                    ]}
        _save(manifest_path, manifest)
        try:
            selection, candidates = _selected_files(source, runtime_files)
            manifest["selection"] = selection
            manifest["runtime_files"] = sorted(runtime_files)
            for relative, skip, tracked in candidates:
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
                    if path.is_file():
                        safe = _safe_file(source, relative)
                        manifest["excluded_files"][relative.as_posix()] = _inspect_file(safe)[0]
                    continue
                path = _safe_file(source, relative)
                before = path.stat()
                sha, private = _inspect_file(path)
                if private:
                    manifest["excluded"].append({"path": relative.as_posix(), "reason": "private_key_content"})
                    manifest["excluded_files"][relative.as_posix()] = sha
                    continue
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied_sha, copied_private = _inspect_file(target)
                after = path.stat()
                if copied_private:
                    # This file is newly created inside this snapshot; never remove source data.
                    target.resolve().relative_to(root)
                    target.unlink()
                    raise SnapshotError(f"Source changed to secret content during capture: {relative}")
                if (copied_sha != sha or before.st_size != after.st_size
                        or before.st_mtime_ns != after.st_mtime_ns):
                    raise SnapshotError(f"Source changed during capture: {relative}")
                manifest["files"][relative.as_posix()] = sha
            digest = hashlib.sha256(json.dumps({"files": manifest["files"],
                                               "excluded_files": manifest["excluded_files"]}, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
            manifest.update(status="captured", digest=digest, file_count=len(manifest["files"]))
            _save(manifest_path, manifest)
            return cls(root, dict(manifest["files"]), digest, manifest_path, dict(manifest["excluded_files"]))
        except Exception as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _save(manifest_path, manifest)
            raise

    @contextmanager
    def environment_like(self, environment, artifact_root, *, read_only=False):
        """Preserve the execution backend; own and close only the new container."""
        if isinstance(environment, DockerEnvironment):
            child = DockerEnvironment(
                self.root, environment.image, f"agent-snapshot-{uuid.uuid4().hex}",
                timeout=environment.timeout, artifact_root=artifact_root,
                **environment.clone_options(),
            )
            child.repo_read_only = read_only
            child.root_read_only = read_only
            try:
                child.start()
                yield child
            finally:
                child.close()
        elif isinstance(environment, LocalEnvironment):
            child = LocalEnvironment(self.root, timeout=environment.timeout, artifact_root=artifact_root)
            child.python_executable = environment.python_executable
            yield child
        else:
            raise TypeError("Snapshot supports LocalEnvironment and DockerEnvironment backends only.")

    def verify_integrity(self):
        """Check source files and newly created source paths; ignore derived caches."""
        current, errors = {}, []
        try:
            if not self.root.is_dir() or _link(self.root):
                raise SnapshotError("Source snapshot root is missing or replaced by a link.")
            for relative, skip in _walk(self.root, preserve_files=self.files):
                reason = _reason(relative)
                if skip or (relative.as_posix() not in self.files and reason and reason not in _SECRET_REASONS):
                    continue
                path = _safe_file(self.root, relative)
                sha, private = _inspect_file(path)
                current[relative.as_posix()] = sha
                if private:
                    errors.append(f"Private key content appeared: {relative}")
        except (OSError, SnapshotError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        change = _changes(self.files, current)
        return {"valid": not any(change.values()) and not errors, **change, "errors": errors,
                "digest": self.digest}

    def changes_from(self, base):
        return _changes(base.files, self.files)

    def excluded_changes_from(self, base):
        """Changes omitted from executable copies cannot be claimed as verified."""
        return _changes(base.excluded_files, self.excluded_files)

    def write_patch(self, base, path, max_text_bytes=1024 * 1024):
        """Write UTF-8 textual changes; manifest explicitly lists omitted changes.

        Binary/non-UTF-8/large files are retained in the snapshot, not embedded in
        the patch. File permission changes and empty directories are not patches.
        """
        for snapshot in (base, self):
            if not snapshot.verify_integrity()["valid"]:
                raise SnapshotError("Source snapshot changed; refusing to export an unverified patch.")
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
                if any(item.exists() and item.stat().st_size > max_text_bytes for item in (before, after)):
                    raise ValueError("file exceeds textual patch size limit")
                old = before.read_bytes() if relative in base.files else b""
                new = after.read_bytes() if relative in self.files else b""
                if b"\0" in old or b"\0" in new:
                    raise ValueError("binary file")
                old_text, new_text = old.decode("utf-8"), new.decode("utf-8")
                if not old and not new:
                    raise ValueError("empty file addition/deletion is not represented by unified text diff")
                diff = difflib.unified_diff(old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
                    fromfile="a/" + relative if relative in base.files else "/dev/null",
                    tofile="b/" + relative if relative in self.files else "/dev/null")
                for line in diff:
                    lines.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
            except (UnicodeError, ValueError) as exc:
                omitted.append({"path": relative, "reason": str(exc)})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes("".join(lines).encode("utf-8"))
        report = {"base_digest": base.digest, "candidate_digest": self.digest, "changes": changes,
                  "patch": str(path), "omitted": omitted, "excluded_changes": excluded_changes,
                  "complete_text_patch": not omitted and not any(excluded_changes.values()),
                  "limitations": ["File modes and empty directories are not represented."]}
        _save(path.with_suffix(path.suffix + ".manifest.json"), report)
        return report


def _changes(before, after):
    return {"added": sorted(after.keys() - before.keys()),
            "deleted": sorted(before.keys() - after.keys()),
            "modified": sorted(path for path in before.keys() & after.keys() if before[path] != after[path])}
