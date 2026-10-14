"""恢复现有 TaskRunner，不替换其规划或执行策略。"""

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from agent.settings import PROJECT_ROOT
import time
from agent.source.snapshots import SourceSnapshot
from agent.source.versions import SourceVersions
from agent.execution.executor import OperationExecutor
from agent.model.tools import SCHEMA_PATH


# ========================================================================
# 执行环境与源码版本身份
# ========================================================================
def environment_key(environment):
    return {
        'backend': type(environment).__name__,
        'repo': str(environment.repo_root.resolve()),
        'artifact_root': str(environment.artifact_root.resolve()),
        'image': getattr(environment, 'image', None),
        'container': getattr(environment, '_container_id', None),
        'python': environment.python_executable,
        'runtime': getattr(environment, 'runtime', None),
        'runtime_files': sorted(getattr(environment, 'runtime_files', ())),
        'runtime_materials_sha256': getattr(environment, 'runtime_materials_sha256', None),
    }


def code_key(root=None):
    names = (
        'agent/__init__.py',
        'agent/__main__.py',
        'agent/acceptance/__init__.py',
        'agent/acceptance/artifact_checks.py',
        'agent/acceptance/behavior.py',
        'agent/acceptance/c_driver.py',
        'agent/acceptance/drivers.py',
        'agent/acceptance/oracles.py',
        'agent/cli/__init__.py',
        'agent/cli/workspace.py',
        'agent/execution/__init__.py',
        'agent/execution/local.py',
        'agent/model/__init__.py',
        'agent/model/deepseek.py',
        'agent/model/network.py',
        'agent/model/payload.py',
        'agent/server/__init__.py',
        'agent/task/control.py',
        'agent/storage/requests.py',
        'agent/storage/resources.py',
        'agent/storage/disk_space.py',
        'agent/application/work_bridge.py',
        'agent/source/__init__.py',
        'agent/source/errors.py',
        'agent/source/artifacts.py',
        'agent/storage/__init__.py',
        'agent/task/__init__.py',
        'agent/task/budget.py',
        'requirements-web.txt',
        'requirements.lock',
        'requirements.txt',
        'resources/plan.schema.json',
        'agent/version.py',
        'agent/server/access.py',
        'agent/execution/public_http.py',
        'agent/execution/projects.py',
        'agent/execution/task_docker.py',
        'agent/execution/preparation.py',
        'agent/execution/managed.py',
        'agent/execution/managed_plan.py',
        'agent/execution/installation_materials.py',
        'agent/execution/managed_proxy.py',
        'agent/execution/managed_setup.py',
        'agent/execution/runtime_inventory.py',
        'agent/cli/interactive.py',
        'agent/cli/tasks.py',
        'agent/model/streaming.py',
        'agent/task/runner.py',
        'agent/task/dialogue.py',
        'agent/task/output_limit.py',
        'resources/prompts/conversation.txt',
        'agent/task/operations.py',
        'agent/storage/preferences.py',
        'agent/settings.py',
        'agent/execution/docker.py',
        'agent/server/workspaces.py',
        'agent/execution/web.py',
        'resources/web.schema.json',
        'agent/server/assistant.py',
        'agent/execution/assistant_files.py',
        'agent/execution/existing.py',
        'agent/execution/process.py',
        'resources/assistant.schema.json',
        'agent/storage/usage.py',
        'agent/storage/permissions.py',
        'agent/execution/desktop.py',
        'agent/server/desktop.py',
        'agent/execution/applications.py',
        'launcher/Applications.cs',
        'launcher/FilePicker.cs',
        'agent/execution/file_actions.py',
        'agent/storage/read_sources.py',
        'agent/source/syntax.py',
        'agent/source/excerpts.py',
        'agent/acceptance/results.py',
        'agent/execution/executor.py',
        'agent/task/contracts.py',
        'agent/model/tools.py',
        'agent/task/repair.py',
        'resources/repair.schema.json',
        'agent/acceptance/revisions.py',
        'resources/acceptance-review.schema.json',
        'agent/acceptance/contracts.py',
        'agent/acceptance/runner.py',
        'resources/drivers/unittest.py',
        'agent/source/snapshots.py',
        'resources/drivers/repository.py',
        'agent/task/instructions.py',
        'agent/model/context.py',
        'agent/model/compaction.py',
        'agent/model/client.py',
        'agent/model/errors.py',
        'resources/context.json',
        'resources/prompts/compaction.txt',
        'agent/source/paging.py',
        'agent/execution/probe.py',
        'agent/source/exploration.py',
        'agent/storage/recovery.py',
        'agent/storage/history.py',
        'agent/storage/product.py',
        'agent/storage/activity.py',
        'agent/storage/live_reasoning.py',
        'agent/execution/probe_batch.py',
        'agent/source/preparation.py',
        'agent/source/probe_preparation.py',
        'agent/source/lifecycle.py',
        'agent/source/view.py',
        'agent/source/versions.py',
        'agent/source/index.py',
        'agent/storage/records.py',
        'agent/acceptance/final.py',
        'resources/prompts/task.txt',
        'resources/prompts/artifact_acceptance.txt',
    )
    schema = SCHEMA_PATH if root is None else Path(root) / SCHEMA_PATH.relative_to(PROJECT_ROOT)
    root = PROJECT_ROOT if root is None else Path(root)
    return hashlib.sha256(
        schema.read_bytes() + b"".join((root / n).read_bytes() for n in names)
    ).hexdigest()


# ========================================================================
# 检查点校验与恢复
# ========================================================================
def restore(loop, stored, identity):
    if stored.get('hard_interruption') or (loop.root / 'hard-interruption.json').exists():
        return conflict(loop, stored, 'Operation was forcefully interrupted; inspect retained evidence before continuing')
    if stored.get('identity') != identity or stored.get('code_key') != code_key():
        raise ValueError('Task or engine changed; stored plan was not executed')
    if stored.get('environment_key') != environment_key(loop.environment):
        return conflict(loop, stored, 'Execution environment changed')

    # ========================================================================
    # 恢复累计预算，并验证原始快照与冻结验收
    # ========================================================================
    restore_budget(loop, stored)
    base = json.loads(Path(stored['baseline']).read_text(encoding='utf-8'))
    loop.baseline = SourceSnapshot(
        Path(base['root']),
        base['files'],
        base['digest'],
        Path(stored['baseline']),
        base.get('excluded_files', {}),
    )
    if not loop.baseline.verify_integrity()['valid']:
        raise ValueError('Original source snapshot changed')
    loop.evidence = SourceVersions(
        loop.environment, loop.root / 'source-records', baseline=loop.baseline
    )
    if loop.evidence.current != stored['source']:
        return conflict(loop, stored, 'Workspace source changed')
    if stored.get('verification_spec'):
        contract = json.loads((loop.root / 'frozen-acceptance.json').read_text(encoding='utf-8'))
        if (
            contract['spec'] != stored['verification_spec']
            or contract['regression'] != stored['regression_command']
        ):
            raise ValueError('Frozen acceptance was changed')
    loop.executor = OperationExecutor(
        loop.environment, loop.evidence, loop.root, versioned_reads=loop.optimize_source
    )
    loop.executor.archive = loop.history_store
    loop.state, loop.rules = stored, stored['rules']
    from agent.acceptance import revisions

    revisions.validate(loop)
    loop.history_store.stats['resumed'] += 1

    # ========================================================================
    # 任何执行结果不明的操作都停在原记录，不自动重放
    # ========================================================================
    if stored.get('assistant_running'):
        stored.update(
            status='execution_unknown', reason='Interrupted assistant operation; no automatic retry'
        )
        return loop.result()
    if stored.get('web_running') or stored.get('web_request_pending'):
        stored.update(
            status='web_request_unknown', reason='Interrupted web request; no automatic retry'
        )
        return loop.result()
    if stored.get('request_pending'):
        reserved = stored['request_pending']['reservation']
        loop.budget.unknown_usage_steps += 1
        loop.budget.unknown_reserved_tokens += reserved
        loop.budget.inflight_reserved_tokens = max(
            0, loop.budget.inflight_reserved_tokens - reserved
        )
        stored.update(
            status='model_request_unknown',
            reason='Interrupted request; no automatic resend',
            request_pending=None,
        )
        return loop.result()
    if (
        stored.get('running_step')
        or stored.get('verification_running')
        or stored.get('acceptance_running')
    ):
        stored.update(
            status='execution_unknown',
            reason='Interrupted execution; reconcile before repeating effects',
        )
        return loop.result()
    if (stored.get('compaction_pending') or {}).get('stage') == 'committing':
        stored.update(
            status='compaction_commit_unknown',
            reason='Interrupted compaction commit; reconcile before continuing',
        )
        return loop.result()

    # ========================================================================
    # 复用已验证结果，或从明确可恢复的状态继续
    # ========================================================================
    from agent.task.output_limit import resume_validated

    resume_validated(loop)
    if stored['status'] == 'verified_success':
        loop.reused_result = True
        loop.history_store.stats['direct_returns'] += 1
        return loop.result()
    if loop.budget.cancelled:
        stored.update(status='task_cancelled', reason=loop.budget.cancel_reason)
        return loop.result()
    if stored['status'] in {'paused', 'awaiting_user'}:
        if not getattr(loop, 'resume_without_input', True):
            return loop.result()
        stored['status'] = 'executing' if stored.get('pending') else 'needs_plan'
    loop.persist('task_resumed', {'completed_steps_not_replayed': True})
    return loop._drive()


# ========================================================================
# 预算策略与累计消耗恢复
# ========================================================================
def restore_budget(loop, stored):
    policy = (
        'max_steps',
        'max_tokens',
        'timeout_seconds',
        'report_reserve',
        'report_token_reserve',
        'report_time_reserve',
        'transport_retry_limit',
    )
    if any(stored['budget'][k] != getattr(loop.budget, k) for k in policy):
        raise ValueError('Budget policy changed; resume cannot reset consumption')
    original_cancel = loop.budget.cancelled
    for item in fields(loop.budget):
        if item.name != 'started_at':
            setattr(loop.budget, item.name, stored['budget'][item.name])
    loop.budget.cancelled |= original_cancel
    loop.prior_elapsed = stored.get('elapsed_seconds', 0)
    loop.budget.started_at = time.monotonic() - loop.prior_elapsed
    loop.calls_before = loop.budget.used_steps


# ========================================================================
# 状态冲突的只读结果
# ========================================================================
def conflict(loop, stored, reason):
    return {
        'status': 'state_conflict',
        'stop_reason': reason,
        'messages': [],
        'rounds': [],
        'verification': {},
        'artifact_dir': str(loop.root),
        'planner_calls': stored['budget']['used_steps'],
        'calls_this_invocation': 0,
        'executor_model_calls': 0,
        'verifier_model_calls': 0,
        'fresh_verification': False,
        'result_reused': False,
    }


# ========================================================================
# 任务目录、独占租约与资源生命周期
# ========================================================================
def run(loop, history, text, budget, verify_command, acceptance):
    from agent.storage.history import LocalHistory
    from agent.task.runner import digest
    from langchain_core.messages import messages_to_dict
    import uuid

    loop.budget = budget
    loop.public_history = list(history)
    if budget.max_tokens is None:
        budget.max_tokens = loop.task_token_limit
    if budget.timeout_seconds is None:
        budget.timeout_seconds = loop.task_timeout
    if getattr(loop, 'preparation', None) is not None and not loop.preparation.closed:
        raise RuntimeError('Previous preparation has not been closed')
    loop.preparation, loop.preparation_errors = None, []
    loop.steps_this_invocation = 0
    loop.calls_before = budget.used_steps
    loop.run_started, loop.prior_elapsed = time.monotonic(), 0
    loop.reused_result, loop.fresh_verification = False, False

    # ========================================================================
    # 固定任务目录、请求身份与源码指纹
    # ========================================================================
    loop.root = (
        loop.explicit_state or loop.artifacts_root / ('task-' + uuid.uuid4().hex)
    ).resolve()
    if loop.root.is_relative_to(loop.environment.repo_root.resolve()) or loop.root.is_relative_to(
        loop.environment.artifact_root.resolve()
    ):
        raise ValueError('History must be outside source and execution mounts')
    loop.identity = digest(
        [
            str(loop.environment.repo_root.resolve()),
            text,
            acceptance,
            verify_command,
            messages_to_dict(history),
        ]
    )
    loop.env_key, loop.code_key = environment_key(loop.environment), code_key()
    loop.history_store = LocalHistory(loop.root) if loop.persistence else None

    # ========================================================================
    # 分别处理非持久模式与持有独占租约的恢复模式
    # ========================================================================
    if not loop.persistence:
        try:
            return loop._run_new(history, text, budget, verify_command, acceptance)
        finally:
            from agent.source.lifecycle import close

            close(loop)
    try:
        with loop.history_store.lease():
            stored = loop.history_store.load()
            return (
                restore(loop, stored, loop.identity)
                if stored
                else loop._run_new(history, text, budget, verify_command, acceptance)
            )
    finally:
        from agent.source.lifecycle import close

        try:
            close(loop)
        finally:
            loop.history_store.close()
