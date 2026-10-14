"""以参数数组运行仓库已有入口，不接受生成的代码或内联代码。"""

import hashlib
import json
from pathlib import Path
import re
import shutil

from agent.execution.assistant_files import AssistantFiles
from agent.execution.process import log_excerpt


def command_for(workspace, entry, args, environment=None):
    if not workspace:
        raise ValueError('运行已有任务需要选择工作目录')
    if len(args) > 100 or any('\0' in v or '\n' in v or '\r' in v for v in args):
        raise ValueError('args 必须是最多 100 个单行参数，不是 Shell 正文')
    files = AssistantFiles(workspace)
    root = files.workspace
    base, separator, target = entry.partition('#')
    if base.startswith('module:'):
        module = base[7:]
        if not re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*', module):
            raise ValueError('module: 后必须是仓库已有模块名')
        relative = Path(*module.split('.'))
        path = files.path(
            str(
                relative.with_suffix('.py')
                if (root / relative.with_suffix('.py')).is_file()
                else relative / '__main__.py'
            )
        )
        prefix = ['-m', module]
    else:
        path = files.path(base)
        prefix = [
            (
                environment.host_to_container(path)
                if environment
                else '/testbed/' + path.relative_to(root).as_posix()
            )
        ]
    raw = files.bytes(path)
    suffix = path.suffix.lower()
    if separator:
        if (
            path.name != 'package.json'
            or not target
            or target not in json.loads(raw).get('scripts', {})
        ):
            raise ValueError('使用 package.json#已有脚本名称，不接受临时任务正文')
        # 严格调用选定的已有脚本，兼容嵌套项目和 Node 22 以前的版本。
        # 关闭自动运行的 pre/post 生命周期钩子。
        directory = (
            environment.host_to_container(path.parent)
            if environment
            else '/testbed/' + path.parent.relative_to(root).as_posix()
        )
        directory = directory.removesuffix('/.')
        executable, prefix = '/usr/bin/env', [
            'npm',
            '--ignore-scripts',
            '--prefix',
            directory,
            'run',
            target,
            '--',
        ]
    elif suffix == '.py':
        executable = (
            environment.python_executable
            if environment
            else '/opt/miniconda3/envs/testbed/bin/python'
        )
    elif suffix in {'.js', '.mjs', '.cjs'}:
        executable, prefix = '/usr/bin/env', ['node', *prefix]
    elif suffix == '.ps1':
        executable, prefix = '/usr/bin/env', [
            'pwsh',
            '-NoProfile',
            '-NonInteractive',
            '-File',
            *prefix,
        ]
    elif suffix == '.sh':
        executable = '/bin/bash'
    else:
        raise ValueError(
            '仅运行已有 .py/.js/.mjs/.cjs/.ps1/.sh 或 package.json#任务；不启动普通应用'
        )
    if not executable:
        raise ValueError('所需运行环境未安装；不会自动安装依赖')
    return [executable, *prefix, *args], path, hashlib.sha256(raw).hexdigest()


def execute(loop, args, folder):
    # 创建容器前先检查入口；生成代码和内联代码始终被拒绝。
    command_for(loop.repo, args['entry'], args['args'])
    loop.activate()
    backend = loop.environment
    command, entry, sha = command_for(loop.repo, args['entry'], args['args'], backend)
    from agent.storage.recovery import environment_key

    loop.state['environment_key'] = environment_key(backend)
    loop.persist('existing_environment_frozen')
    timeout = args['timeout']
    if not 1 <= timeout <= loop.task_timeout:
        raise ValueError('timeout 必须在 1..任务超时秒数之间')
    timeout = min(timeout, loop.budget.remaining_time() or timeout)
    previous_usage = None
    if args['usage_file']:
        # 报告文件可能尚未生成；只检查预定位置，不额外授予目录权限。
        report = Path(args['usage_file'])
        if report.is_absolute() or '..' in report.parts or ':' in args['usage_file']:
            raise ValueError('usage_file 必须是工作目录内的相对路径，或留空')
        candidate = Path(loop.repo) / report
        if candidate.exists():
            candidate = AssistantFiles(loop.repo).path(args['usage_file'])
            previous_usage = (
                candidate.stat().st_mtime_ns,
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
            )

    def started(pid):
        loop.state['assistant_process'] = {
            'container_id': pid,
            'entry': args['entry'],
            'entry_sha256': sha,
        }
        loop.persist('existing_process_started', loop.state['assistant_process'])

    outputs = backend.artifact_root / 'existing' / folder.name
    outputs.mkdir(parents=True, exist_ok=True)
    result = backend.run_process(
        command,
        stdout_path=outputs / 'stdout.log',
        stderr_path=outputs / 'stderr.log',
        timeout=timeout,
        on_started=started,
    )
    for name in ('stdout.log', 'stderr.log'):
        shutil.copy2(outputs / name, folder / name)
    if result.get('error') and result.get('exit_code') is None:
        result['status'] = 'unknown'
    result.update(
        entry=args['entry'],
        entry_sha256=sha,
        command=command,
        stdout=log_excerpt(folder / 'stdout.log'),
        stderr=log_excerpt(folder / 'stderr.log'),
        notice='运行结果不是编码验收；失败只报告，不自动修改源码。',
    )
    result['child_usage'] = {'status': 'unknown', 'source': args['usage_file'] or None}
    if args['usage_file']:
        try:
            path = AssistantFiles(loop.repo).path(args['usage_file'])
            raw = AssistantFiles(loop.repo).bytes(path)
            if previous_usage == (path.stat().st_mtime_ns, hashlib.sha256(raw).hexdigest()):
                raise ValueError('用量文件没有本次生成或更新的证据，不沿用旧成本')
            usage = json.loads(raw)
            keys = ('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens')
            if not isinstance(usage, dict) or not all(
                type(usage.get(k)) is int and usage[k] >= 0 for k in keys
            ):
                raise ValueError('用量记录需包含非负的 input/output/total/cached_input_tokens')
            if (
                usage['total_tokens'] != usage['input_tokens'] + usage['output_tokens']
                or usage['cached_input_tokens'] > usage['input_tokens']
            ):
                raise ValueError('用量记录的合计不一致')
            (folder / 'child-usage.json').write_bytes(raw)
            result['child_usage'] = {
                'status': 'reported_by_task',
                'source': str(path),
                'sha256': hashlib.sha256(raw).hexdigest(),
                **{k: usage[k] for k in keys},
            }
        except (ValueError, OSError) as exc:
            result['child_usage']['detail'] = str(exc)
    return result
