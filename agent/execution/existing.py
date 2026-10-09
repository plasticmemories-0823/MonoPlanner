"""Run repository-owned entries as argv; never accept generated or inline code."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from agent.execution.assistant_files import AssistantFiles
from agent.execution.process import run_process, log_excerpt


def command_for(workspace, entry, args):
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
        path = files.path(str(relative.with_suffix('.py') if (root/relative.with_suffix('.py')).is_file()
                              else relative/'__main__.py'))
        prefix = ['-m', module]
    else:
        path = files.path(base)
        prefix = [str(path)]
    raw = files.bytes(path)
    suffix = path.suffix.lower()
    if separator:
        if path.name != 'package.json' or not target or target not in json.loads(raw).get('scripts', {}):
            raise ValueError('使用 package.json#已有脚本名称，不接受临时任务正文')
        # Node 22 --run dispatches only an existing package script; no npm installation.
        executable, prefix = shutil.which('node'), ['--run', target, '--']
    elif suffix == '.py':
        candidates = [root/'.venv/Scripts/python.exe', root/'.venv/bin/python',
                      root/'venv/Scripts/python.exe', root/'venv/bin/python']
        executable = str(next((p for p in candidates if p.is_file()), Path(sys.executable)))
    elif suffix in {'.js', '.mjs', '.cjs'}:
        executable = shutil.which('node')
    elif suffix == '.ps1':
        executable, prefix = shutil.which('pwsh') or shutil.which('powershell'), ['-NoProfile', '-NonInteractive', '-File', str(path)]
    elif suffix == '.sh':
        executable = shutil.which('bash')
    else:
        raise ValueError('仅运行已有 .py/.js/.mjs/.cjs/.ps1/.sh 或 package.json#任务；不启动普通应用')
    if not executable:
        raise ValueError('所需运行环境未安装；不会自动安装依赖')
    return [executable, *prefix, *args], path, hashlib.sha256(raw).hexdigest()


def execute(loop, args, folder):
    command, entry, sha = command_for(loop.repo, args['entry'], args['args'])
    timeout = args['timeout']
    if not 1 <= timeout <= loop.task_timeout:
        raise ValueError('timeout 必须在 1..任务超时秒数之间')
    timeout = min(timeout, loop.budget.remaining_time() or timeout)
    previous_usage = None
    if args['usage_file']:
        # A report may not exist yet. Check its intended location without granting another root.
        report = Path(args['usage_file'])
        if report.is_absolute() or '..' in report.parts or ':' in args['usage_file']:
            raise ValueError('usage_file 必须是工作目录内的相对路径，或留空')
        candidate = Path(loop.repo)/report
        if candidate.exists():
            candidate = AssistantFiles(loop.repo).path(args['usage_file'])
            previous_usage = (candidate.stat().st_mtime_ns, hashlib.sha256(candidate.read_bytes()).hexdigest())
    env = {k: v for k, v in os.environ.items() if not re.search(r'(API_KEY|ACCESS_TOKEN|SECRET|PASSWORD)', k, re.I)}
    env.update(PYTHONUNBUFFERED='1', PYTHONPATH=str(Path(loop.repo).resolve()))

    def started(pid):
        loop.state['assistant_process'] = {'pid': pid, 'entry': args['entry'], 'entry_sha256': sha}
        loop.persist('existing_process_started', loop.state['assistant_process'])

    result = run_process(command, cwd=loop.repo, stdout_path=folder/'stdout.log', stderr_path=folder/'stderr.log',
                         timeout=timeout, env=env, on_started=started)
    result.update(entry=args['entry'], entry_sha256=sha, command=command,
                  stdout=log_excerpt(folder/'stdout.log'), stderr=log_excerpt(folder/'stderr.log'),
                  notice='运行结果不是编码验收；失败只报告，不自动修改源码。')
    result['child_usage'] = {'status': 'unknown', 'source': args['usage_file'] or None}
    if args['usage_file']:
        try:
            path = AssistantFiles(loop.repo).path(args['usage_file'])
            raw = AssistantFiles(loop.repo).bytes(path)
            if previous_usage == (path.stat().st_mtime_ns, hashlib.sha256(raw).hexdigest()):
                raise ValueError('用量文件没有本次生成或更新的证据，不沿用旧成本')
            usage = json.loads(raw)
            keys = ('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens')
            if not isinstance(usage, dict) or not all(type(usage.get(k)) is int and usage[k] >= 0 for k in keys):
                raise ValueError('用量记录需包含非负的 input/output/total/cached_input_tokens')
            if usage['total_tokens'] != usage['input_tokens'] + usage['output_tokens'] or usage['cached_input_tokens'] > usage['input_tokens']:
                raise ValueError('用量记录的合计不一致')
            (folder/'child-usage.json').write_bytes(raw)
            result['child_usage'] = {'status': 'reported_by_task', 'source': str(path),
                                     'sha256': hashlib.sha256(raw).hexdigest(), **{k: usage[k] for k in keys}}
        except (ValueError, OSError) as exc:
            result['child_usage']['detail'] = str(exc)
    return result
