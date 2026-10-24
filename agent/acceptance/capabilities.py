"""开工前公开验收适配边界，不用模型猜测某个目标能否执行。"""

import json
from pathlib import Path


def wine_environment(repo, root=None):
    """只读取明确选择的镜像声明；不运行命令或替用户安装环境。"""
    if not repo or not Path(repo).is_dir():
        return {'available': False, 'reason': 'Select an existing configured Wine project directory.'}
    from agent.execution.projects import workspace_binding, docker_query

    try:
        binding = workspace_binding(repo, root=root)
        image = binding.get('image')
        if not image:
            from agent.execution.managed import ManagedEnvironments
            prepared = ManagedEnvironments(root).status(repo)
            if prepared.get('state') == 'ready':
                image = prepared.get('image')
        if not image:
            return {'available': False, 'reason': 'Wine CMD is not ready. Ask the user to open Current project > Project environment '
                      '> Install Wine. Only that user action installs it; do not install it with work tools.'}
        inspected = json.loads(docker_query(['image', 'inspect', image]))[0]
        labels = inspected.get('Config', {}).get('Labels') or {}
        available = (inspected['Id'] == image and inspected['Os'] == 'linux'
                     and inspected['Architecture'] == 'amd64'
                     and labels.get('org.monoplanner.wine-cmd') == '1')
        return {'available': available, 'image': image,
                'evidence': 'Local immutable image and Wine CMD adapter declaration; '
                            'the actual interpreter and artifact are checked during execution.',
                'reason': '' if available else 'The selected image does not declare Wine CMD protocol 1.'}
    except (OSError, ValueError, KeyError, IndexError) as error:
        return {'available': False, 'reason': 'Cannot verify the selected Wine image.',
                'error_type': type(error).__name__}


def describe(*, repo=None, root=None):
    return {
        'execution_environment': 'Linux Docker; no host shell fallback',
        'repository_entry': ['python', 'node', 'shell', 'wine-cmd'],
        'python_invocation': {
            'modes': ['file', 'module (-m)'],
            'module_binding': 'A repository module .py or regular package __main__.py; '
                              'parent packages must be regular repository packages, not namespace packages.',
            'import_scope': 'The repository import root is added only to the executed project child. '
                            'Compilation uses isolated Python and does not import the project.',
        },
        'generated_code': {'c': ['numeric expression', 'numeric assignment']},
        'python_api': {
            'entry': 'An original repository module:function or module:Class.method. When the '
                     'original request specifies a whole library rather than an exact API, '
                     'module:* binds that original module/package scope without inventing a function.',
            'verification': 'Standard unittest methods must actually call the synchronous Python '
                            'implementation. Merely importing or replacing it is not execution.',
        },
        'file_artifact': {
            'entry': 'The original relative output path, not an invented producer script name.',
            'verification': 'Read unchanged delivered bytes inside the bound unittest method with '
                            'acceptance_artifacts.read_artifact(relative_path), then parse/load and '
                            'assert the required behavior. The test may not create its deliverable.',
            'maximum_bytes': 512 * 1024 * 1024,
        },
        'wine_cmd': {
            **wine_environment(repo, root),
            'scope': 'BAT/CMD runs in Wine compatibility mode. CMD has no compile-only mode; '
                     'only executed paths are parsed and tested. This does not validate native '
                     'Windows services, registry, permissions or real system-cleanup effects.',
            'configuration': {'installer': 'user_button_only',
                              'ui_path': '当前项目 → 项目执行环境 → 安装 Wine',
                              'project_scope': 'selected existing project directory',
                              'starts_work': False},
            'literal_argument_limits': 'No control characters or % ! " & | < > ^ in entry paths/argv.',
        },
        'unavailable': ['native Windows CMD/BAT', 'Windows services and registry'],
        'unverified_coding_handoff': False,
        'notice': 'Adapter support is not proof of installed project dependencies. '
                  'The confirmed environment must still pass preparation before execution. '
                  'Supporting documentation belongs in the delivery behavior, not as a runnable entry.',
    }


def require_targets(targets, *, repo=None, root=None):
    for index, target in enumerate(targets):
        supported = ({'c'} if target['artifact_type'] == 'generated_code' else
                     {'python'} if target['artifact_type'] in {'python_api', 'file_artifact'} else
                     {'python', 'node', 'shell', 'wine-cmd'})
        if target['target_runtime'] not in supported:
            raise ValueError(
                f"targets[{index}].target_runtime={target['target_runtime']!r} has no supported "
                'execution adapter. Work has not been started. Explain the capability gap and '
                'required environment to the user before proposing confirmation. Do not change '
                'their requested artifact or promise that continuing can fix a missing runtime. '
                'Supporting text documents must not be declared as executable entries.'
            )
        if target['target_runtime'] == 'wine-cmd':
            from resources.drivers.execution import wine_command
            wine_command('/testbed/' + target['entry'], target['argv'])
    if any(target['target_runtime'] == 'wine-cmd' for target in targets):
        capability = wine_environment(repo, root)
        if not capability['available']:
            raise ValueError('Wine CMD acceptance is unavailable for this project: '
                             + capability['reason'] + ' No work has started. '
                             'Ask the user to select this project directory and click Install Wine in Project environment. '
                             'Do not start a coding task to install Wine or issue installation commands.')
