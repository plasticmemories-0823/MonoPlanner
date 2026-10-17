"""RAG 调参操作入口；只消费已有评测记录，不生成回答或下载模型。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from agent.rag.models import LocalModels, ModelUnavailable
from agent.rag.tuning.controller import TuningController
from agent.rag.tuning.laya import LayaAdapter
from agent.rag.tuning.policy import Config, canonical, enumerate_candidates
from agent.rag.types import RagConfig


def _read(path, limit=4 * 1024 * 1024):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('The input record exceeds its size limit')
    return json.loads(raw.decode('utf-8-sig'))


def _parser():
    parser = argparse.ArgumentParser(
        description='Operate bounded RAG tuning from recorded evidence.',
    )
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--initial-config', type=Path)
    parser.add_argument('--output', type=Path,
                        help='Write a new JSON record; never overwrite a file.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('status')
    propose = commands.add_parser('propose')
    propose.add_argument('--metrics', type=Path, required=True)
    shadow = commands.add_parser('shadow')
    shadow.add_argument('--proposal', type=Path, required=True)
    shadow.add_argument('--spec', type=Path, required=True)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--trial', required=True)
    evaluate.add_argument('--rows', type=Path, required=True)
    evaluate.add_argument('--qualification', type=Path, required=True)
    evaluate.add_argument('--phase', choices=('shadow', 'canary'), default='shadow')
    canary = commands.add_parser('canary')
    canary.add_argument('--trial', required=True)
    rollback = commands.add_parser('rollback')
    rollback.add_argument('--trial', required=True)
    rollback.add_argument('--reason', required=True)
    return parser


def _operate(args, models_factory, clock):
    if args.command == 'status' and not args.db.is_file():
        return {'initialized': False, 'paid_generation_started': False}
    initial = None
    if not args.db.is_file():
        if args.command != 'propose':
            raise ValueError('Initialize tuning with propose before changing trial state')
        values = _read(args.initial_config) if args.initial_config else asdict(RagConfig())
        initial = Config.create(values)
    controller = TuningController(args.db, initial, clock)
    if args.command == 'status':
        return controller.snapshot()
    if args.command == 'propose':
        saved = controller.snapshot()['baseline']
        base = Config.create(saved['values'], saved['parent'])
        candidates = enumerate_candidates(base)
        # 短标签避免 SHA-256 编号挤占 Laya 的选项 Token 预算。
        indexed = {'no_change': candidates[0]}
        indexed.update({f'c{index}': item for index, item in enumerate(candidates[1:], 1)})
        options = {'no_change': 'Keep the current configuration.'}
        options.update({key: f'{item.knob}: {base.values[item.knob]} -> '
                        f'{item.config.values[item.knob]}'
                        for key, item in indexed.items() if item.knob is not None})
        metrics = _read(args.metrics, limit=16 * 1024)
        if not isinstance(metrics, dict):
            raise ValueError('Tuning metrics must be a JSON object')
        adapter = LayaAdapter(models_factory(args.models).laya_provider())
        decision = adapter.choose({'metrics': metrics}, options)
        output = {'schema': 1, 'status': decision.status, 'reason': decision.reason,
                  'judge_identity': decision.identity, 'base_version': base.version,
                  'paid_generation_started': False}
        if decision.status == 'ready':
            chosen = indexed[decision.choice]
            output.update(choice=decision.choice, knob=chosen.knob,
                          candidate_version=chosen.config.version,
                          values=chosen.config.values, parent=chosen.config.parent)
        return output
    if args.command == 'shadow':
        proposal = _read(args.proposal)
        if proposal.get('schema') != 1 or proposal.get('status') != 'ready':
            raise ValueError('A complete recorded candidate proposal is required')
        baseline = controller.snapshot()['baseline']['version']
        if proposal.get('base_version') != baseline:
            raise ValueError('The proposal was made against another baseline')
        candidate = Config.create(proposal['values'], proposal['parent'])
        if candidate.version != proposal.get('candidate_version'):
            raise ValueError('Candidate proposal fingerprint mismatch')
        trial = controller.start_shadow(candidate, _read(args.spec))
        return {'status': controller.trial_status(trial), 'trial': trial,
                'paid_generation_started': False}
    if args.command == 'evaluate':
        try:
            rows, qualification = _read(args.rows), _read(args.qualification)
            if not isinstance(rows, list) or not isinstance(qualification, dict):
                raise ValueError('Provide paired result rows and a human qualification record')
            method = (controller.review_canary if args.phase == 'canary'
                      else controller.evaluate_shadow)
            return method(args.trial, rows, qualification)
        except Exception as error:
            if args.phase == 'canary':
                controller.rollback(args.trial, f'invalid_canary_record:{type(error).__name__}')
            raise
    if args.command == 'canary':
        identity = models_factory(args.models).laya_provider().identity
        controller.activate_canary(args.trial, identity)
        return {'status': 'canary', 'trial': args.trial, 'new_task_percentage': 5}
    changed = controller.rollback(args.trial, args.reason)
    return {'status': 'rolled_back' if changed else 'unchanged', 'trial': args.trial}


def main(argv=None, *, models_factory=LocalModels, clock=time.time):
    args = _parser().parse_args(argv)
    try:
        if args.output and args.output.exists():
            raise ValueError('Output already exists; use a new record path')
        result = _operate(args, models_factory, clock)
        encoded = canonical(result)
        if args.output:
            with args.output.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(encoded + '\n')
        print(encoded)
        if result.get('status') in {'unavailable', 'invalid'} or result.get('passed') is False:
            return 2
        return 0
    except (ValueError, OSError, KeyError, TypeError, ModelUnavailable) as error:
        print(canonical({'status': 'error', 'error': str(error), 'paid_generation_started': False}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
