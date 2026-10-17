"""冻结评测、持久化灰度与回滚；模型建议不能自行通过门禁。"""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import time
from typing import Any, Callable, Mapping, Sequence

from .policy import Config, canonical, fingerprint, validate_candidate


QUALITY = ('faithfulness', 'citation_precision')
RESOURCES = ('latency_ms', 'memory_bytes', 'cost')
LABELS = frozenset({'supported', 'contradicted', 'unknown'})
GIB = 1024 ** 3


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in '0123456789abcdef' for character in value
    )


def freeze_spec(value: Mapping[str, Any]) -> dict:
    """在试验前固定所有判据；缺少数据或判据不能推断为通过。"""
    spec = json.loads(canonical(dict(value)))
    defaults = {
        'bootstrap_samples': 1000, 'confidence': 0.95, 'seed': 1729,
        'qualification_min_cases': 100, 'qualification_min_accuracy': 0.9,
        'max_trials': 1,
    }
    for key, default in defaults.items():
        spec.setdefault(key, default)
    queries = spec.get('queries')
    if not isinstance(queries, dict) or not 100 <= len(queries) <= 200:
        raise ValueError('A frozen evaluation requires 100 to 200 labeled queries')
    if any(not key or not isinstance(tags, list) or
           any(not isinstance(tag, str) for tag in tags) for key, tags in queries.items()):
        raise ValueError('Frozen query IDs and tags must be valid strings')
    required = {'hard_case', 'no_answer', 'permission_boundary'}
    if not required.issubset({tag for tags in queries.values() for tag in tags}):
        raise ValueError('Frozen evaluation is missing required query strata')
    optimization_ids = spec.get('optimization_ids')
    if not isinstance(optimization_ids, list) or not optimization_ids or any(
        not isinstance(item, str) or not item for item in optimization_ids
    ) or set(optimization_ids).intersection(queries):
        raise ValueError('Optimization and held-out query IDs must be nonempty and disjoint')
    if spec.get('primary_metric') not in {'recall_at_100', 'ndcg'}:
        raise ValueError('Freeze recall_at_100 or ndcg as the primary metric')
    if not _finite(spec.get('min_effect')) or not 0 < spec['min_effect'] < 1:
        raise ValueError('Freeze a positive minimum meaningful improvement before evaluation')
    for name in ('dataset_sha256', 'qualification_dataset_sha256'):
        if not _sha(spec.get(name)):
            raise ValueError(f'{name} must identify the frozen labeled dataset')
    for name in ('judge_identity', 'generation_identity'):
        identity = spec.get(name)
        if not isinstance(identity, dict) or not identity or any(
            not isinstance(item, str) or not item for item in identity.values()
        ):
            raise ValueError(f'{name} must be fixed before evaluation')
    for name, low, high in (('bootstrap_samples', 200, 5000),
                            ('qualification_min_cases', 30, 10_000), ('max_trials', 1, 16)):
        if type(spec[name]) is not int or not low <= spec[name] <= high:
            raise ValueError(f'{name} is outside its supported range')
    if not _finite(spec['confidence']) or not 0.9 <= spec['confidence'] <= 0.999:
        raise ValueError('confidence must be between 0.9 and 0.999')
    minimum = spec['qualification_min_accuracy']
    if not _finite(minimum) or not 0.5 < minimum <= 1:
        raise ValueError('qualification_min_accuracy must be greater than 0.5 and at most 1')
    if type(spec['seed']) is not int:
        raise ValueError('The bootstrap seed must be an integer')
    return spec


def qualification_errors(spec: Mapping, record: Mapping | None) -> list[str]:
    if not isinstance(record, Mapping):
        return ['missing_human_qualification']
    if record.get('human_labeled') is not True or (
        record.get('dataset_sha256') != spec['qualification_dataset_sha256']
        or record.get('judge_identity') != spec['judge_identity']
    ):
        return ['qualification_identity_mismatch']
    cases = record.get('cases')
    if not isinstance(cases, list) or len(cases) < spec['qualification_min_cases']:
        return ['insufficient_qualification_cases']
    identifiers = set()
    by_label = {label: [0, 0] for label in LABELS}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get('id'), str) or not case['id']:
            return ['invalid_qualification_case']
        expected, predicted = case.get('expected'), case.get('predicted')
        if case['id'] in identifiers or not isinstance(expected, str) or (
            not isinstance(predicted, str) or expected not in LABELS or predicted not in LABELS
        ):
            return ['invalid_qualification_case']
        identifiers.add(case['id'])
        bucket = by_label[case['expected']]
        bucket[0] += int(case['predicted'] == case['expected'])
        bucket[1] += 1
    if any(count == 0 or correct / count < spec['qualification_min_accuracy']
           for correct, count in by_label.values()):
        return ['qualification_class_accuracy_failed']
    # 单一高频标签不能掩盖失败；总准确率再采用保守的 Wilson 下界。
    size = len(cases)
    probability = sum(correct for correct, _ in by_label.values()) / size
    z = 1.959963984540054
    lower = (probability + z * z / (2 * size) - z * math.sqrt(
        probability * (1 - probability) / size + z * z / (4 * size * size)
    )) / (1 + z * z / size)
    if lower < spec['qualification_min_accuracy']:
        return ['qualification_accuracy_not_established']
    return []


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate_gate(spec: Mapping, rows: Sequence[Mapping], qualification: Mapping | None,
                  baseline_version: str, candidate_version: str) -> dict:
    spec = freeze_spec(spec)
    reasons = qualification_errors(spec, qualification)
    metrics = (spec['primary_metric'],) + QUALITY + RESOURCES
    paired = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get('query_id') not in spec['queries']:
            reasons.append('unexpected_query')
            continue
        query_id = row['query_id']
        if row.get('dataset_sha256') != spec['dataset_sha256']:
            reasons.append(f'{query_id}:dataset_mismatch')
            continue
        if query_id in paired:
            reasons.append('duplicate_query')
            continue
        valid = True
        for side, version in (('baseline', baseline_version), ('candidate', candidate_version)):
            sample = row.get(side)
            if not isinstance(sample, Mapping) or sample.get('version') != version:
                reasons.append(f'{query_id}:{side}:config_mismatch')
                valid = False
                continue
            if sample.get('generator') != spec['generation_identity'] or (
                sample.get('judge_identity') != spec['judge_identity']
            ) or not all(
                _sha(sample.get(key)) for key in ('input_sha256', 'answer_sha256', 'record_sha256')
            ) or sample.get('usage_known') is not True:
                reasons.append(f'{query_id}:{side}:missing_generation_provenance')
                valid = False
            if sample.get('permission_pass') is not True or sample.get('safety_pass') is not True:
                reasons.append(f'{query_id}:{side}:safety_failure')
                valid = False
            for metric in metrics:
                value = sample.get(metric)
                if not _finite(value) or value < 0 or (
                    metric not in RESOURCES and value > 1
                ):
                    reasons.append(f'{query_id}:{side}:invalid_{metric}')
                    valid = False
            if _finite(sample.get('memory_bytes')) and sample['memory_bytes'] > GIB:
                reasons.append(f'{query_id}:{side}:shared_memory_budget_exceeded')
                valid = False
        if valid:
            if row['baseline']['record_sha256'] == row['candidate']['record_sha256']:
                reasons.append(f'{query_id}:candidate_reuses_baseline_generation')
            else:
                paired[query_id] = row
    if set(paired) != set(spec['queries']):
        reasons.append('incomplete_paired_holdout')
    report = {'passed': False, 'reasons': sorted(set(reasons)), 'intervals': {},
              'spec_id': fingerprint(spec), 'rows_sha256': fingerprint(list(rows)),
              'qualification_sha256': fingerprint(qualification)}
    if reasons:
        return report

    ordered = [paired[key] for key in sorted(paired)]
    values = {name: ([row['baseline'][name] for row in ordered],
                     [row['candidate'][name] for row in ordered]) for name in metrics}
    rng = random.Random(spec['seed'])
    distributions = {name: [] for name in metrics}
    size = len(ordered)
    for _ in range(spec['bootstrap_samples']):
        indices = [rng.randrange(size) for _ in range(size)]
        for name, (baseline, candidate) in values.items():
            if name == 'latency_ms':
                difference = _percentile([candidate[index] for index in indices], 0.95)
                difference -= _percentile([baseline[index] for index in indices], 0.95)
            elif name == 'memory_bytes':
                difference = max(candidate[index] for index in indices)
                difference -= max(baseline[index] for index in indices)
            else:
                difference = sum(candidate[index] - baseline[index] for index in indices) / size
            distributions[name].append(difference)

    # 同一次试验的多个门禁和预先声明的候选次数共同分摊错误概率。
    tail = (1 - spec['confidence']) / (2 * len(metrics) * spec['max_trials'])
    for name, differences in distributions.items():
        lower, upper = _percentile(differences, tail), _percentile(differences, 1 - tail)
        report['intervals'][name] = {'lower': lower, 'upper': upper}
        if name == spec['primary_metric'] and lower <= spec['min_effect']:
            report['reasons'].append('primary_improvement_not_established')
        elif name in QUALITY and lower < 0:
            report['reasons'].append(f'{name}_non_regression_not_established')
        elif name in RESOURCES and upper > 0:
            report['reasons'].append(f'{name}_non_regression_not_established')
    report['passed'] = not report['reasons']
    return report


class TuningController:
    """SQLite 事务同时保护配置版本、任务绑定、冷却期和审计记录。"""

    def __init__(self, path: Path, initial: Config | None = None,
                 clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.clock = clock
        if initial is None:
            if not self.path.is_file():
                raise ValueError('The tuning database has not been initialized')
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS configs (
                    version TEXT PRIMARY KEY, payload TEXT NOT NULL, parent TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT);
                CREATE TABLE IF NOT EXISTS trials (
                    id TEXT PRIMARY KEY, base TEXT, candidate TEXT, spec TEXT, status TEXT,
                    report TEXT, evidence TEXT);
                CREATE TABLE IF NOT EXISTS bindings (task_id TEXT PRIMARY KEY, version TEXT);
                CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, at REAL, event TEXT);
            ''')
            self._save_config(connection, initial)
            connection.execute('INSERT OR IGNORE INTO state VALUES (1, ?)', (canonical({
                'baseline': initial.version, 'canary': None, 'trial': None,
                'last_activation': None,
            }),))

    def snapshot(self) -> dict:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            state = self._state(connection)
            baseline = self._config(connection, state['baseline'])
            trials = connection.execute(
                'SELECT id, base, candidate, status FROM trials ORDER BY rowid DESC LIMIT 20',
            ).fetchall()
            return {
                'initialized': True, 'state': state,
                'baseline': {'version': baseline.version, 'parent': baseline.parent,
                             'values': baseline.values},
                'trials': [dict(row) for row in trials],
                'bound_tasks': connection.execute('SELECT COUNT(*) FROM bindings').fetchone()[0],
            }
        finally:
            connection.close()

    def trial_status(self, trial_id: str) -> str:
        with self._transaction() as connection:
            row = connection.execute('SELECT status FROM trials WHERE id=?',
                                     (trial_id,)).fetchone()
            if row is None:
                raise ValueError('Unknown tuning trial')
            return row['status']

    @contextmanager
    def _transaction(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute('BEGIN IMMEDIATE')
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _save_config(connection, config: Config):
        connection.execute('INSERT OR IGNORE INTO configs VALUES (?, ?, ?)',
                           (config.version, config.payload, config.parent))

    @staticmethod
    def _config(connection, version: str) -> Config:
        row = connection.execute('SELECT * FROM configs WHERE version=?', (version,)).fetchone()
        if row is None:
            raise ValueError('Unknown persisted RAG configuration')
        config = Config.create(json.loads(row['payload']), row['parent'])
        if config.version != version:
            raise ValueError('Persisted RAG configuration fingerprint mismatch')
        return config

    @staticmethod
    def _state(connection) -> dict:
        return json.loads(connection.execute('SELECT value FROM state WHERE id=1').fetchone()[0])

    def _record(self, connection, state: dict, event: dict):
        connection.execute('UPDATE state SET value=? WHERE id=1', (canonical(state),))
        connection.execute('INSERT INTO audit(at, event) VALUES (?, ?)',
                           (self.clock(), canonical(event)))

    def start_shadow(self, candidate: Config, spec: Mapping) -> str:
        frozen = freeze_spec(spec)
        with self._transaction() as connection:
            state = self._state(connection)
            baseline = self._config(connection, state['baseline'])
            knob = validate_candidate(baseline, candidate)
            if knob is None:
                raise ValueError('no_change does not create an evaluation trial')
            if state['canary'] is not None:
                raise ValueError('Finish or roll back the current canary before another trial')
            spec_json = canonical(frozen)
            trial_id = fingerprint([baseline.version, candidate.version, frozen])
            existing = connection.execute(
                'SELECT id FROM trials WHERE id=?', (trial_id,),
            ).fetchone()
            if existing is not None:
                return trial_id
            if connection.execute("SELECT id FROM trials WHERE status='shadow'").fetchone():
                raise ValueError('Only one unevaluated shadow candidate is allowed')
            count = connection.execute('SELECT COUNT(*) FROM trials WHERE spec=?',
                                       (spec_json,)).fetchone()[0]
            if count >= frozen['max_trials']:
                raise ValueError('The frozen evaluation candidate budget is exhausted')
            self._save_config(connection, candidate)
            connection.execute('INSERT INTO trials VALUES (?, ?, ?, ?, ?, NULL, NULL)',
                               (trial_id, baseline.version, candidate.version, spec_json, 'shadow'))
            self._record(connection, state, {
                'type': 'shadow_started', 'trial': trial_id, 'knob': knob,
                'before': baseline.values[knob], 'after': candidate.values[knob],
            })
            return trial_id

    def evaluate_shadow(self, trial_id: str, rows: Sequence[Mapping],
                        qualification: Mapping | None) -> dict:
        # 统计计算在事务外执行，任务绑定不会等 bootstrap 完成。
        with self._transaction() as connection:
            trial = connection.execute('SELECT * FROM trials WHERE id=?', (trial_id,)).fetchone()
            if trial is None or trial['status'] != 'shadow':
                raise ValueError('Only an unevaluated shadow trial may accept results')
            trial = dict(trial)
        report = evaluate_gate(json.loads(trial['spec']), rows, qualification,
                               trial['base'], trial['candidate'])
        evidence = canonical({'rows': list(rows), 'qualification': qualification})
        with self._transaction() as connection:
            cursor = connection.execute(
                'UPDATE trials SET status=?, report=?, evidence=? WHERE id=? AND status=?',
                ('shadow_passed' if report['passed'] else 'shadow_failed', canonical(report),
                 evidence, trial_id, 'shadow'),
            )
            if cursor.rowcount != 1:
                raise ValueError('Shadow trial changed while evaluation was running')
            self._record(connection, self._state(connection), {
                'type': 'shadow_evaluated', 'trial': trial_id, 'report': report,
            })
        return report

    def activate_canary(self, trial_id: str, judge_identity: Mapping[str, str]):
        with self._transaction() as connection:
            state = self._state(connection)
            trial = connection.execute('SELECT * FROM trials WHERE id=?', (trial_id,)).fetchone()
            if trial is None or trial['status'] != 'shadow_passed':
                raise ValueError('A passing shadow evaluation is required before canary activation')
            if dict(judge_identity) != json.loads(trial['spec'])['judge_identity']:
                raise ValueError('The qualified judge identity changed before activation')
            if state['canary'] is not None or trial['base'] != state['baseline']:
                raise ValueError('The baseline or active canary changed')
            now = self.clock()
            if state['last_activation'] is not None and now - state['last_activation'] < 86_400:
                raise ValueError('At most one effective change is allowed every 24 hours')
            state.update(canary=trial['candidate'], trial=trial_id, last_activation=now)
            connection.execute("UPDATE trials SET status='canary' WHERE id=?", (trial_id,))
            self._record(connection, state, {'type': 'canary_started', 'trial': trial_id,
                                             'percentage': 5})

    def bind_task(self, task_id: str) -> Config:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError('A nonempty task ID is required for configuration binding')
        with self._transaction() as connection:
            existing = connection.execute('SELECT version FROM bindings WHERE task_id=?',
                                          (task_id,)).fetchone()
            if existing is not None:
                return self._config(connection, existing['version'])
            state = self._state(connection)
            version = state['baseline']
            if state['canary'] is not None:
                digest = hashlib.sha256(canonical([state['trial'], task_id]).encode()).digest()
                if int.from_bytes(digest[:8], 'big') % 100 < 5:
                    version = state['canary']
            connection.execute('INSERT INTO bindings VALUES (?, ?)', (task_id, version))
            self._record(connection, state, {'type': 'task_bound', 'task': task_id,
                                             'version': version})
            return self._config(connection, version)

    def rollback(self, trial_id: str, reason: str) -> bool:
        if not reason:
            raise ValueError('Rollback requires an explicit diagnostic reason')
        with self._transaction() as connection:
            state = self._state(connection)
            if state['trial'] != trial_id or state['canary'] is None:
                return False
            state.update(canary=None, trial=None)
            connection.execute("UPDATE trials SET status='rolled_back' WHERE id=?", (trial_id,))
            self._record(connection, state, {'type': 'rolled_back', 'trial': trial_id,
                                             'reason': reason})
            return True

    def review_canary(self, trial_id: str, rows: Sequence[Mapping],
                      qualification: Mapping | None) -> dict:
        with self._transaction() as connection:
            state = self._state(connection)
            trial = connection.execute('SELECT * FROM trials WHERE id=?', (trial_id,)).fetchone()
            if trial is None or state['trial'] != trial_id or trial['status'] != 'canary':
                raise ValueError('Canary review does not match the active trial')
            trial = dict(trial)
        try:
            report = evaluate_gate(json.loads(trial['spec']), rows, qualification,
                                   trial['base'], trial['candidate'])
        except Exception as error:
            # 不完整或非法遥测不能让异常版本继续接收新任务。
            self.rollback(trial_id, f'invalid_canary_evaluation:{type(error).__name__}')
            raise
        if not report['passed']:
            self.rollback(trial_id, ';'.join(report['reasons']))
        with self._transaction() as connection:
            self._record(connection, self._state(connection), {
                'type': 'canary_reviewed', 'trial': trial_id, 'report': report,
            })
        return report
