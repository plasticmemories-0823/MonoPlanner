"""将固定源码字节与可选准备流程接入现有 TaskRunner。"""

from agent.model.errors import ModelRequestNotSent
from agent.source.preparation import Preparation


def collect(loop):
    prep = getattr(loop, 'preparation', None)
    if prep is None:
        return
    try:
        ready = prep.collect(loop.evidence.current, loop.history_store)
        loop.executor.prepared = ready
        if ready is not None and (
            loop.executor.navigator is None or loop.executor.navigator.version != ready['version']
        ):
            from agent.source.index import RepositoryNavigator

            loop.executor.navigator = RepositoryNavigator.from_prepared(
                loop.evidence.view(), ready, loop.root / 'source-records' / 'navigation'
            )
    except Exception as exc:
        loop.preparation_errors.append(type(exc).__name__)
        loop.executor.prepared = None


def begin(loop):
    if not loop.optimize_source or not loop.overlap_preparation:
        return
    if loop.preparation is None:
        from agent.execution.docker import DockerEnvironment

        loop.preparation = Preparation(
            loop.root / 'source-records' / 'prepared',
            prepare_probes=isinstance(loop.environment, DockerEnvironment),
        )
        if loop.preparation.probes is not None:
            loop.executor.prepared_probe = loop.preparation.probes.take
    collect(loop)
    loop.preparation.start(loop.evidence.view())


def close(loop):
    prep = getattr(loop, 'preparation', None)
    if prep is not None and not prep.closed:
        try:
            prep.close(loop.evidence.current, loop.history_store)
        except Exception as exc:
            loop.preparation_errors.append(type(exc).__name__)
    if hasattr(loop, 'state') and loop.state:
        loop.state['source_performance'] = {
            'optimized': loop.optimize_source,
            'prepared_line_hits': getattr(getattr(loop, 'executor', None), 'prepared_line_hits', 0),
            'preparation': prep.report() if prep else None,
            'preparation_errors': list(loop.preparation_errors),
        }


def boundary(loop, kind, **fields):
    expected = loop.state['source']
    actual = loop.evidence.record(kind, **fields)
    if actual != expected:
        raise ModelRequestNotSent(
            'state_conflict',
            'Workspace changed since the recorded evidence; no planned write or acceptance ran.',
        )
    return actual


def readonly(loop, plan, step):
    return loop.optimize_source and (step.op in {'read', 'search'} or plan.kind == 'inspect')


def configure_lookups(loop):
    from agent.model.compaction import history
    from agent.source.paging import visible_reads
    from agent.source.exploration import ExplorationMap

    loop.executor.visible_reads = visible_reads(
        history(loop.state, loop.root), loop.state['pending']['rows']
    )
    from pathlib import Path

    for (name, sha), spans in list(loop.executor.visible_reads.items()):
        path = Path(name)
        if path.is_absolute() and path.is_relative_to(loop.environment.repo_root):
            loop.executor.visible_reads[
                (path.relative_to(loop.environment.repo_root).as_posix(), sha)
            ] = spans
    if not hasattr(loop.executor, 'exploration'):
        loop.executor.exploration = ExplorationMap(
            loop.executor, loop.state.setdefault('source_map', {})
        )
    loop.executor.exploration.sync()


def record_lookup(loop, step, row, directory):
    if row['status'] != 'succeeded' or step.op not in {'read', 'search'}:
        return
    from agent.storage.records import save

    result = loop.executor.exploration.observe(row['data'])
    path = directory / 'repository-map.json'
    save(path, {'source_version': result['source_version'], 'files': loop.state['source_map']})
    if loop.history_store:
        loop.history_store.register_derived(path, 'source_navigation', result['source_version'])
    result['record'] = '$record/' + path.relative_to(loop.root).as_posix()
    row['data']['repository_map'] = result
