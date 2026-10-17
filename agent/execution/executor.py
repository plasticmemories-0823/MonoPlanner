"""执行提交的读取、搜索和 Shell 操作，不进行模型决策。"""

from contextlib import contextmanager
import hashlib
import json
import re
import time
from pathlib import Path, PurePosixPath
from agent.source.snapshots import _safe_file
from agent.storage.records import clip


class OperationExecutor:
    """只运行提交的操作，不自行作出新的语义决策。"""

    def __init__(self, environment, evidence, records, *, versioned_reads=False):
        self.environment, self.evidence = environment, evidence
        self.root = Path(records).resolve()
        self.versioned_reads = versioned_reads
        self.prepared = None
        self.prepared_probe = None
        self.prepared_line_hits = 0
        self.navigator = None
        self.visible_reads = {}

    @contextmanager
    def probe_batch(self, directory, enabled=True):
        from agent.execution.probe_batch import ProbeBatch

        if getattr(self, "_probe_batch", None) is not None:
            raise RuntimeError("Nested probe batches are not allowed")
        batch = (
            ProbeBatch(
                self.environment,
                directory,
                source_factory=self.evidence.view if self.versioned_reads else None,
                prepared_source=self.prepared_probe,
            )
            if enabled
            else None
        )
        self._probe_batch = batch
        try:
            yield
        finally:
            self._probe_batch = None
            if batch is not None:
                batch.close()

    def inventory(self):
        if self.versioned_reads:
            return self.evidence.view().files
        path = self.evidence.root / "states" / (self.evidence.current + ".json")
        return json.loads(path.read_text(encoding="utf-8"))["files"]

    def path(self, name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or ":" in name:
            raise ValueError("Use a relative path inside the bound workspace")
        if name.startswith("$record/"):
            relative = Path(name[8:])
            if not relative.parts or relative.parts[0] not in {
                "plans",
                "requests",
                "feedback",
                "drafts",
                "acceptance",
                "tool-payloads",
            }:
                raise ValueError("Only archived operation evidence is readable")
            base = self.root
        else:
            if relative.as_posix() not in self.inventory():
                raise ValueError("Path is absent or excluded from the source inventory")
            base = self.environment.repo_root
        path = _safe_file(base, relative)
        from agent.model.tool_payloads import readable

        payload = readable(name, getattr(self, 'archive', None))
        if name.startswith('$record/tool-payloads/') and not payload:
            raise ValueError('Tool payload paths require an exact SHA-256 filename.')
        if not payload and path.stat().st_size > 2_000_000:
            raise ValueError("Read a bounded source/log file (at most 2MB)")
        return path

    def source_bytes(self, name):
        raw = (
            self.evidence.view().read_bytes(name)
            if self.versioned_reads
            else self.path(name).read_bytes()
        )
        if len(raw) > 2_000_000:
            raise ValueError('Read a bounded source/log file (at most 2MB)')
        return raw

    def source_lines(self, name):
        if self.prepared and self.prepared['version'] == self.evidence.current:
            lines = self.prepared['lines'].get(name)
            if lines is not None:
                self.prepared_line_hits += 1
                return lines
        return self.source_bytes(name).decode('utf-8').splitlines()

    def read(self, name, offset, limit, *, budget=None):
        name = PurePosixPath(name.replace('\\', '/')).as_posix()
        from agent.model.tool_payloads import read, readable

        archive = getattr(self, 'archive', None)
        if readable(name, archive):
            return read(self.root, name, offset, limit, budget=budget, archive=archive)
        if name.startswith('$record/'):
            data = self.path(name).read_bytes()
            lines = data.decode('utf-8').splitlines()
        else:
            data = self.source_bytes(name)
            lines = self.source_lines(name)
        from agent.source.paging import ReadBudget

        budget = budget if budget is not None else ReadBudget()
        return {
            "path": name,
            "offset": offset,
            "total_lines": len(lines),
            "sha256": hashlib.sha256(data).hexdigest(),
            **budget.excerpt(lines, offset, limit),
        }

    def has_visible_overlap(self, step):
        if step.path.startswith('$') or step.query.startswith('reread:'):
            return False
        name = PurePosixPath(step.path.replace('\\', '/')).as_posix()
        sha = self.inventory().get(name)
        return any(
            start < step.offset + step.limit and end > step.offset
            for start, end in self.visible_reads.get((name, sha), [])
        )

    def read_continuation(self, path, offset, limit, repeat=False, *, budget=None):
        from agent.source.paging import ranges, unread, read_pages, read_range, ReadBudget

        budget = budget if budget is not None else ReadBudget()
        ranges(offset, limit)  # 即使不需要补读，也必须检查参数限制。
        path = PurePosixPath(path.replace('\\', '/')).as_posix()
        if path.startswith('$record/'):
            from agent.model.tool_payloads import readable

            data = (self.read(path, offset, limit, budget=budget)
                    if readable(path, getattr(self, 'archive', None))
                    else read_range(self, path, offset, limit, budget))
            pages = list(read_pages(data))
            data['continuation'] = {
                'requested_offset': offset,
                'requested_lines': limit,
                'next_offset': pages[-1]['next_offset'],
                'reason': 'archived_record',
            }
            return data
        sha = self.inventory().get(path)
        total = len(self.source_lines(path))  # 已经可见的原文区间仍需校验访问权限。
        end = min(total, offset + limit)
        seen = self.visible_reads.get((path, sha), []) if not repeat else []
        missing = unread(offset, end, seen)
        if missing == [[offset, end]] or offset >= total:
            data = read_range(self, path, offset, limit, budget)
        elif missing:
            reads = []
            for start, stop in missing:
                reads.append(read_range(self, path, start, stop - start, budget))
                if reads[-1]['truncated']:
                    break
            data = (
                reads[0]
                if len(reads) == 1
                else {'reads': reads, 'truncated': any(r['truncated'] for r in reads)}
            )
        else:
            data = {
                'path': path,
                'sha256': sha,
                'offset': offset,
                'total_lines': total,
                'text': '',
                'visible_ranges': [],
                'already_visible': True,
                'truncated': False,
            }
        returned = [pair for page in read_pages(data) for pair in page['visible_ranges']]
        from agent.source.paging import merged

        self.visible_reads[(path, sha)] = merged(
            [*self.visible_reads.get((path, sha), []), *returned]
        )
        gaps = unread(offset, end, [*seen, *returned])
        data['continuation'] = {
            'requested_offset': offset,
            'requested_lines': limit,
            'reused_ranges': [
                [max(offset, start), min(end, stop)]
                for start, stop in seen
                if start < end and stop > offset
            ],
            'next_offset': gaps[0][0] if gaps else end,
            'reason': (
                'explicit_reread'
                if repeat
                else 'unseen_ranges' if seen else 'no_matching_visible_source'
            ),
        }
        return data

    def search(self, step):
        if step.path == "$record" or step.path.startswith("$record/"):
            archive = getattr(self, "archive", None)
            if archive is None:
                raise ValueError("Indexed history is disabled; use an exact $record path")
            items = archive.find(
                step.query,
                min(step.limit, 50),
                step.path[8:] if step.path.startswith("$record/") else "",
            )
            return {
                "hits": [
                    {
                        "path": "$record/" + v["path"],
                        "line": 1,
                        "text": v["kind"] + ": " + (v["query"] or ""),
                    }
                    for v in items
                ],
                "truncated": len(items) >= min(step.limit, 50),
                "files_examined": None,
            }
        prefix = Path(step.path).as_posix().strip("/")
        if prefix in {"", "."}:
            prefix = ""
        if Path(step.path).is_absolute() or ".." in Path(step.path).parts or ":" in step.path:
            raise ValueError("Search scope must be a repository-relative directory")
        hits, examined = [], 0
        names = sorted(
            n for n in self.inventory() if not prefix or n.startswith(prefix + "/") or n == prefix
        )
        for name in names:
            examined += 1
            try:
                lines = self.source_lines(name)
            except UnicodeError:
                continue
            for i, line in enumerate(lines):
                if step.query in line:
                    hits.append({"path": name, "line": i + 1, "text": line[:500]})
                    if len(hits) >= min(step.limit, 50):
                        return {"hits": hits, "truncated": True, "files_examined": examined}
        return {"hits": hits, "truncated": False, "files_examined": examined}

    def execute(self, step, prior, directory, timeout, *, isolated=False):
        started = time.monotonic()
        if step.op == "read":
            from agent.source.paging import ReadBudget, OUTPUT_BYTES

            budget = ReadBudget()
            if step.query.startswith('payload_bytes:'):
                from agent.model.tool_payloads import read_bytes

                match = re.fullmatch(r'payload_bytes:(0|[1-9][0-9]*)', step.query)
                if not match or step.from_step or step.offset != 0 or step.limit != 2000:
                    raise ValueError('Payload byte reads require an exact path, offset=0, default '
                                     'limit=2000 and query=payload_bytes:<byte offset>; the shared '
                                     '50 KiB display budget bounds this mode, not the line limit.')
                data = read_bytes(
                    self.root, step.path.replace('\\', '/'), int(match[1]),
                    budget=budget, archive=getattr(self, 'archive', None),
                )
            elif step.from_step:
                from agent.model.tool_results import normalize_result

                source = prior[step.from_step]
                result = normalize_result(
                    "search_files", source["data"], source_version=source.get("before"),
                    project=self.environment.repo_root,
                )
                hits = result["hits"][:5]
                if not hits:
                    raise ValueError("Referenced search has no hits; no path was guessed")
                data = {
                    "reads": [
                        self.read_continuation(
                            h["path"],
                            max(0, h["line"] - 1 - step.limit // 2),
                            step.limit,
                            step.query.startswith('reread:'),
                            budget=budget,
                        )
                        for h in hits
                    ]
                }
                data['truncated'] = any(r['truncated'] for r in data['reads'])
            else:
                data = self.read_continuation(
                    step.path,
                    step.offset,
                    step.limit,
                    step.query.startswith('reread:'),
                    budget=budget,
                )
            data['output_budget'] = {
                'limit_bytes': OUTPUT_BYTES,
                'used_bytes': OUTPUT_BYTES - budget.remaining,
                'encoding': 'utf-8',
                'scope': 'read_step_text',
            }
            ok = True
        elif step.op == "search":
            data, ok = self.search(step), True
            data.update(
                query=step.query,
                scope=step.path,
                requested_limit=step.limit,
                match_mode="literal_case_sensitive",
            )
        elif isolated:
            from agent.execution.probe import run_probe

            data = run_probe(
                self.environment,
                step.command,
                directory,
                timeout,
                batch=getattr(self, "_probe_batch", None),
                source_version=self.evidence.current,
            )
            ok = data.get("exit_code") in step.expected_exit and not (
                data.get("error") or data.get("timed_out")
            )
            for key in ("stdout", "stderr"):
                data[key] = clip(data.get(key, ""))
                data[key + "_record"] = (
                    (Path(directory) / (key + ".log")).relative_to(self.root).as_posix()
                )
        else:
            logs = (
                self.environment.artifact_root
                / "plan-steps"
                / self.root.name
                / Path(directory).name
            )
            logs.mkdir(parents=True, exist_ok=False)
            stdout, stderr = logs / "stdout.log", logs / "stderr.log"
            data = self.environment.run_process(
                step.command, stdout_path=stdout, stderr_path=stderr, timeout=timeout, shell=True
            )
            complete = data.get("exit_code") is not None and not (
                data.get("timed_out") or data.get("error")
            )
            ok = complete and data["exit_code"] in step.expected_exit
            for key, path in (("stdout", stdout), ("stderr", stderr)):
                raw = path.read_bytes() if path.exists() else b""
                destination = Path(directory) / (key + ".log")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
                data[key] = clip(raw.decode("utf-8", errors="replace"))
                data[key + "_record"] = destination.relative_to(self.root).as_posix()
            data["expected_exit"] = step.expected_exit
        return {
            "status": "succeeded" if ok else "failed",
            "data": data,
            "seconds": time.monotonic() - started,
        }
