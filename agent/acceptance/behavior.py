"""Declaration-only numeric behavior checks within the existing final acceptance."""
import ast
import hashlib
import json
import time
from agent.acceptance.artifact_checks import run_artifact_checks


def declaration_only(code):
    """Never silently discard an existing unittest or other executable statement."""
    try:
        body = ast.parse(code).body
    except (SyntaxError, TypeError):
        return False
    return (len(body) == 1 and isinstance(body[0], ast.Assign)
            and len(body[0].targets) == 1
            and isinstance(body[0].targets[0], ast.Name)
            and body[0].targets[0].id == 'ARTIFACT_CHECKS')


def run_behavior_contract(owner, phase, report, started):
    """Only real compiler/consumer results can mark this contract as passed."""
    directory = owner.artifact_dir / (phase + '.artifacts')
    try:
        if not owner._unchanged():
            raise ValueError('Frozen acceptance files changed before execution')
        remaining = owner.timeout - (time.monotonic() - started)
        if remaining <= 0 or not owner.artifact_checks:
            raise RuntimeError('No time or no declared behavior checks remain')
        backend = owner.environment
        rows = run_artifact_checks(owner.artifact_checks, backend, directory, remaining)
        ids = tuple(sorted(row['id'] for row in rows))
        if owner._test_ids is not None and ids != owner._test_ids:
            raise ValueError('Frozen behavior check IDs changed')
        if owner._test_ids is None:
            owner._test_ids = ids
            inventory = owner.artifact_dir / 'test_inventory.json'
            inventory.write_text(json.dumps(ids), encoding='utf-8')
            owner._hashes[inventory] = hashlib.sha256(inventory.read_bytes()).digest()
        errors = sum(row['status'] == 'error' for row in rows)
        failures = sum(row['status'] == 'failed' for row in rows)
        passed = bool(rows) and all(row['status'] == 'passed' for row in rows)
        status = 'passed' if passed else 'error' if errors else 'failed'
        report.update(status=status, exit_code=0 if passed else 1,
            reason='Declared raw-output behavior passed.' if passed else 'Raw-output compilation or behavior did not pass.',
            tests_run=sum(row['status'] != 'not_run' for row in rows),
            failures=failures, errors=errors, skipped=0,
            expected_failures=0, unexpected_successes=0,
            test_ids=list(ids), test_results=rows, artifact_results=rows,
            test_evidence_valid=True, error_kind='result_error' if errors else None,
            acceptance_mode='artifact_behavior_only')
    except (OSError, RuntimeError, ValueError) as exc:
        report.update(status='error', error_kind='result_error', test_evidence_valid=False,
                      reason='Behavior acceptance could not complete: ' + str(exc))
    if not owner._unchanged():
        report.update(status='error', integrity_ok=False, error_kind='integrity_error',
                      test_evidence_valid=False, reason='Frozen acceptance was modified')
    for key in ('stdout', 'stderr'):
        log = directory / (key + '.log')
        if log.is_file():
            report[key] = owner._summary(log)
            report[key + '_log'] = str(log)
    report['seconds'] = time.monotonic() - started
    return report
