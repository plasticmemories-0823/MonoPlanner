"""按计划延迟创建诊断沙箱，不参与模型决策或验收判断。"""

import base64
import json
import time
import uuid
from pathlib import Path

from agent.execution.docker import DockerEnvironment
from agent.source.snapshots import SourceSnapshot


class ProbeBatch:
    """只在一次计划执行期间复用同一份不可变源码和容器。"""

    def __init__(self, environment, directory, source_factory=None, prepared_source=None):
        self.environment = environment
        self.source_factory = source_factory
        self.prepared_source = prepared_source
        self.directory = Path(directory)
        self.child = self.snapshot = None
        self.version = None
        self.closed = False
        self.stats = {}

    def _start(self, version, timeout):
        started = time.monotonic()
        self.folder = self.directory / ('session-' + uuid.uuid4().hex)
        self.folder.mkdir(parents=True)
        self.snapshot = self.prepared_source(version) if self.prepared_source is not None else None
        prepared = self.snapshot is not None
        self.snapshot = self.snapshot or (
            self.source_factory(version).export(self.folder / 'source')
            if self.source_factory is not None
            else SourceSnapshot.capture(
                self.environment.repo_root,
                self.folder / 'source',
                excluded_roots=(self.environment.artifact_root,),
                runtime_files=getattr(self.environment, 'runtime_files', ()),
            )
        )
        self.runtime = self.folder / 'runtime'
        self.child = DockerEnvironment(
            self.snapshot.root,
            self.environment.image,
            'agent-probe-' + uuid.uuid4().hex,
            timeout=timeout,
            artifact_root=self.runtime,
            **self.environment.clone_options(),
        )
        self.child.repo_read_only = True
        self.child.root_read_only = True
        self.stats = {
            'source_version': version,
            'source_digest': self.snapshot.digest,
            'commands': 0,
            'command_seconds': 0.0,
            'prepared_source': prepared,
            'source_seconds': time.monotonic() - started,
        }
        try:
            self.child.start()
        except BaseException:
            self._finish_session()
            raise
        self.version = version
        self.stats['setup_seconds'] = time.monotonic() - started
        self._save_stats()

    def _save_stats(self):
        (self.folder / 'session.json').write_text(
            json.dumps(self.stats, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    def _finish_session(self):
        child, self.child = self.child, None
        if child is None:
            return
        started = time.monotonic()
        try:
            child.close()
        finally:
            self.stats['teardown_seconds'] = time.monotonic() - started
            self._save_stats()

    def close(self):
        self.closed = True
        self._finish_session()

    def run(self, command, directory, timeout, source_version=None):
        from agent.execution.probe import _PROBE_RUNNER, MAX_OUTPUT_BYTES, MAX_PROBE_SECONDS

        if self.closed:
            raise RuntimeError('A closed probe batch cannot be reused')
        if not isinstance(self.environment, DockerEnvironment):
            return {
                'exit_code': None,
                'timed_out': False,
                'executed': False,
                'mode': 'isolated_probe',
                'error_kind': 'probe_backend_unavailable',
                'error': 'Isolated probes require Docker; no local command ran.',
            }
        if timeout <= 0:
            raise ValueError('Probe timeout must be positive')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.child is not None and self.version != source_version:
            self._finish_session()
        if self.child is None:
            self._start(source_version, min(timeout, MAX_PROBE_SECONDS))
        self.stats['commands'] += 1
        logs = self.runtime / ('command-%04d' % self.stats['commands'])
        stdout, stderr = logs / 'stdout.log', logs / 'stderr.log'
        payload = base64.b64encode(
            json.dumps(
                {'command': command, 'max_output': MAX_OUTPUT_BYTES}, ensure_ascii=False
            ).encode('utf-8')
        ).decode('ascii')
        data, started = {}, time.monotonic()
        try:
            data = self.child.run_process(
                [self.child.python_executable, '-c', _PROBE_RUNNER, payload],
                stdout_path=stdout,
                stderr_path=stderr,
                timeout=min(timeout, MAX_PROBE_SECONDS),
                shell=False,
            )
        except BaseException:
            self._finish_session()
            raise
        finally:
            for name, source in (('stdout', stdout), ('stderr', stderr)):
                if source.is_symlink() or not source.resolve().is_relative_to(
                    self.runtime.resolve()
                ):
                    self._finish_session()
                    raise ValueError('Probe log escaped its own runtime')
                raw = source.read_bytes() if source.exists() else b''
                (directory / (name + '.log')).write_bytes(raw)
                data[name] = raw.decode('utf-8', errors='replace')
            self.stats['command_seconds'] += time.monotonic() - started
            self._save_stats()
        integrity = self.snapshot.verify_integrity()
        data.update(
            mode='isolated_probe',
            executed=True,
            source_read_only=True,
            snapshot_digest=self.snapshot.digest,
            snapshot_integrity=integrity,
            acceptance=False,
            network='none',
        )
        if not integrity['valid']:
            data.update(
                error='Read-only probe source integrity failed',
                error_kind='probe_integrity_error',
                exit_code=None,
            )
        if data.get('exit_code') == 125 and 'PROBE_OUTPUT_LIMIT_EXCEEDED' in data['stderr']:
            data.update(
                error='Probe output exceeded its bound; partial logs retained',
                error_kind='probe_output_limit',
                output_limited=True,
            )
        data['meaning'] = 'Observation from an isolated copy; no acceptance or workspace edits.'
        if data.get('error') or data.get('timed_out') or data.get('exit_code') != 0:
            self._finish_session()
        return data
