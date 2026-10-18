"""将固定源码字节与可选准备流程接入现有 TaskRunner。"""

from agent.model.errors import ModelRequestNotSent
from agent.source.preparation import Preparation
from agent.source.errors import source_storage_boundary


def collect(loop):
    prep = getattr(loop, 'preparation', None)
    if prep is None:
        return
    try:
        ready = prep.collect(loop.evidence.current, loop.history_store)
        if ready is not None and ready is not loop.executor.prepared:
            view = loop.evidence.view()
            if ready['version'] != view.version or any(
                view.files.get(name) != entry['digest']
                for name, entry in ready['entries'].items()
            ):
                raise ValueError('Prepared navigation does not match the recorded source')
        loop.executor.prepared = ready
    except Exception as exc:
        loop.preparation_errors.append(type(exc).__name__)
        loop.executor.prepared = None
        if isinstance(exc, OSError):
            raise


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
            if isinstance(exc, OSError):
                raise
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
    from agent.rag.container_sources import is_container_path
    from agent.source.paging import visible_reads
    from agent.source.exploration import ExplorationMap

    def source_identity(page):
        if is_container_path(page['path']):
            # 相同正文在不同容器绑定中也不能继承可见行；旧无版本记录不用于跳读。
            return (page['path'], page['version']) if page.get('version') else None
        return page['path'], page['sha256']

    loop.executor.visible_reads = visible_reads(
        history(loop.state, loop.root), loop.state['pending']['rows'], key=source_identity,
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
            loop.executor, loop.state.setdefault('source_map', {}),
            loop.state.setdefault('source_map_coverage', {}),
        )
    loop.executor.exploration.sync()


@source_storage_boundary
def record_lookup(loop, step, row, directory):
    if row['status'] != 'succeeded' or step.op not in {'read', 'search'}:
        return
    from agent.rag.container_sources import is_container_path
    from agent.source.paging import read_pages
    from agent.storage.records import save

    paths = [page['path'] for page in read_pages(row['data'])]
    paths.extend(hit['path'] for hit in row['data'].get('hits', []))
    if ((paths and all(is_container_path(path) for path in paths))
            or (not paths and is_container_path(getattr(step, 'path', '')))):
        # 容器原件由只读适配器归档，不以项目版本生成误导性的仓库映射。
        return
    path = directory / 'repository-map.json'
    record = '$record/' + path.relative_to(loop.root).as_posix()
    row['data'].pop('repository_map', None)  # 读取缓存不能重发上次的导航投递。
    queued = [item['data'].get('repository_map')
              for item in (loop.state.get('pending') or {}).get('rows', [])
              if item['status'] == 'succeeded' and item['op'] in {'read', 'search'}]
    result = loop.executor.exploration.observe(row['data'], queued=queued, record=record)
    if result is None:
        return
    save(path, {'format': 'repository-map-v3', 'source_version': result['source_version'],
                'coverage': loop.executor.exploration.coverage(),
                'files': {name: loop.state['source_map'][name] for name in paths
                          if name in loop.state['source_map']}})
    if loop.history_store:
        loop.history_store.register_derived(path, 'source_navigation', result['source_version'])
    row['data']['repository_map'] = result
