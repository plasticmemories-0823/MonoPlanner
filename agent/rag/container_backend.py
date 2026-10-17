"""把固定只读协议发送给已有任务容器；不启动环境或接受模型程序。"""

import base64
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess

from agent.model.errors import ModelInterrupted
from agent.rag import container_probe
from agent.rag.types import source_read_limit
from agent.task import control


_CONTAINER_FORMAT = ('{"container_id":{{json .Id}},"image_id":{{json .Image}},'
                     '"running":{{json .State.Running}},"mounts":{{json .Mounts}}}')
_IMAGE_FORMAT = '{"image_id":{{json .Id}},"repo_digests":{{json .RepoDigests}}}'
_SCRIPT = Path(container_probe.__file__).read_text(encoding='utf-8')


class ContainerReadError(ValueError):
    def __init__(self, code, message, **details):
        self.details = {'code': code, 'message': message, **details}
        super().__init__(message)


def _path(value):
    if (not isinstance(value, str) or not value.startswith('/') or len(value) > 4096
            or '\\' in value or '\0' in value or ':' in value
            or '..' in PurePosixPath(value).parts):
        raise ContainerReadError('invalid_path', 'Container sources require absolute POSIX paths.')
    return value


class ContainerBackend:
    def __init__(self, environment):
        self.environment = environment
        timeout = getattr(environment, 'timeout', 30)
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)):
            raise ContainerReadError('invalid_timeout', 'The container timeout is invalid.')
        self.timeout = min(30, timeout)
        if self.timeout <= 0:
            raise ContainerReadError('invalid_timeout', 'The container timeout must be positive.')
        self._identity = None

    def _check(self):
        control.check_interrupted(pause_only=True)
        check = getattr(self.environment, '_check_cancelled', None)
        if check:
            try:
                check()
            except InterruptedError as error:
                raise ModelInterrupted('pause', sent=False) from error

    def _query(self, arguments, maximum):
        self._check()
        try:
            result = self.environment._docker(
                arguments, timeout=self.timeout, capture_output=True, text=True,
                encoding='utf-8', errors='strict',
            )
        except (OSError, subprocess.SubprocessError, UnicodeError) as error:
            self._check()
            code = ('container_query_timeout' if isinstance(error, subprocess.TimeoutExpired)
                    else 'container_query_failed')
            raise ContainerReadError(code, 'The read-only container query did not complete.') from error
        self._check()
        if result.returncode:
            raise ContainerReadError('container_query_failed',
                                     'The existing container did not complete its read-only query.')
        text = result.stdout
        if not isinstance(text, str) or len(text.encode('utf-8')) > maximum:
            raise ContainerReadError('response_too_large', 'The container response exceeded its budget.')
        try:
            return json.loads(text)
        except (TypeError, ValueError) as error:
            raise ContainerReadError('invalid_response', 'The container returned invalid protocol data.') from error

    def _container(self):
        identity = getattr(self.environment, '_container_id', None)
        if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{12,64}', identity):
            raise ContainerReadError('container_unavailable', 'No existing task container is bound.')
        value = self._query(['inspect', '--format', _CONTAINER_FORMAT, identity], 262144)
        if (not isinstance(value, dict) or not isinstance(value.get('container_id'), str)
                or not re.fullmatch('[0-9a-f]{64}', value['container_id'])
                or not value['container_id'].startswith(identity)
                or not re.fullmatch('sha256:[0-9a-f]{64}', str(value.get('image_id', '')))
                or not isinstance(value.get('mounts'), list)):
            raise ContainerReadError('invalid_binding', 'The inspected container binding is invalid.')
        if value.get('running') is not True:
            raise ContainerReadError('container_not_running', 'The bound task container is not running.')
        expected_image = getattr(self.environment, 'image', '')
        if str(expected_image).startswith('sha256:') and value['image_id'] != expected_image:
            raise ContainerReadError('binding_changed', 'The task container image changed.')
        identity = {key: value[key] for key in ('container_id', 'image_id', 'mounts')}
        if self._identity is not None and identity != self._identity:
            raise ContainerReadError('binding_changed', 'The task container or its mounts changed.')
        self._identity = deepcopy(identity)
        return identity

    def binding(self):
        current = self._container()
        image = self._query(['image', 'inspect', '--format', _IMAGE_FORMAT, current['image_id']], 65536)
        if not isinstance(image, dict) or image.get('image_id') != current['image_id']:
            raise ContainerReadError('binding_changed', 'The inspected image identity changed.')
        digests = image.get('repo_digests') or []
        if (not isinstance(digests, list) or len(digests) > 128
                or any(not isinstance(item, str) or len(item) > 1024 for item in digests)):
            raise ContainerReadError('invalid_binding', 'The image digest metadata is invalid.')
        facts = getattr(self.environment, 'runtime_facts', {})
        versions = {key: value for key, value in facts.items()
                    if key in {'python', 'node', 'gcc', 'clang', 'compiler'}
                    and isinstance(value, str) and len(value) <= 4096} if isinstance(facts, dict) else {}
        return {**current, 'repo_digests': sorted(digests), 'toolchain_versions': {
            'status': 'recorded' if versions else 'unavailable',
            'source': 'bound_runtime_facts' if versions else 'not_probed', 'values': versions,
        }}

    def call(self, operation, payload):
        fields = {'stat': set(), 'list': {'limit', 'after'}, 'read': {'max_bytes'}}
        if not isinstance(operation, str) or operation not in fields or not isinstance(payload, dict):
            raise ContainerReadError('invalid_operation', 'Only stat, list, and read are supported.')
        common = {'path', 'allowed_root', 'denied_roots'}
        if not {'path', 'allowed_root'} <= set(payload) or set(payload) - common - fields[operation]:
            raise ContainerReadError('invalid_payload', 'Unsupported container read fields.')
        payload = {**payload, 'denied_roots': payload.get('denied_roots', [])}
        _path(payload['path'])
        _path(payload['allowed_root'])
        if not isinstance(payload['denied_roots'], list) or len(payload['denied_roots']) > 64:
            raise ContainerReadError('invalid_payload', 'The excluded source roots are invalid.')
        for item in payload['denied_roots']:
            _path(item)
        if operation == 'list':
            limit, after = payload.get('limit', 200), payload.get('after', '')
            if (type(limit) is not int or not 1 <= limit <= container_probe.MAX_ENTRIES
                    or not isinstance(after, str) or len(after) > 4096
                    or any(value in after for value in ('/', '\\', '\0')) or after in {'.', '..'}):
                raise ContainerReadError('invalid_cursor', 'The directory page limit or cursor is invalid.')
        maximum = min(source_read_limit(), container_probe.MAX_BYTES)
        if operation == 'read':
            payload.setdefault('max_bytes', maximum)
            if type(payload['max_bytes']) is not int or not 0 <= payload['max_bytes'] <= maximum:
                raise ContainerReadError('invalid_limit', 'The source read exceeds the shared memory budget.')
            maximum = payload['max_bytes']
        request = json.dumps({'operation': operation, 'payload': payload,
                              'deadline_seconds': min(25, self.timeout * 0.8)}, ensure_ascii=True,
                             separators=(',', ':'))
        if len(request.encode('utf-8')) > container_probe.MAX_REQUEST_BYTES:
            raise ContainerReadError('invalid_payload', 'The container read request is too large.')
        current = self._container()
        python = _path(getattr(self.environment, 'python_executable', ''))
        answer = self._query(
            ['exec', current['container_id'], python, '-I', '-S', '-B', '-c', _SCRIPT, request],
            (4 * ((maximum + 2) // 3) + 65536) if operation == 'read'
            else container_probe.MAX_LIST_BYTES + 65536,
        )
        if not isinstance(answer, dict) or type(answer.get('ok')) is not bool:
            raise ContainerReadError('invalid_response', 'The container returned invalid read protocol data.')
        if not answer['ok']:
            error = answer.get('error')
            if not isinstance(error, dict):
                raise ContainerReadError('invalid_response', 'The container returned an invalid failure.')
            raise ContainerReadError(str(error.get('code', 'source_unavailable'))[:128],
                                     str(error.get('message', 'The container source is unavailable.'))[:512])
        result = answer.get('result')
        if not isinstance(result, dict):
            raise ContainerReadError('invalid_response', 'The read protocol result is missing.')
        if operation == 'read':
            try:
                raw = base64.b64decode(result['data_base64'], validate=True)
                valid = (len(raw) <= maximum and len(raw) == result['size']
                         and hashlib.sha256(raw).hexdigest() == result['sha256'])
            except (ValueError, TypeError, KeyError):
                valid = False
            if not valid:
                raise ContainerReadError('invalid_response', 'The original container bytes failed verification.')
        return result
