"""Runtime observations in a fresh Docker sandbox; never candidate acceptance."""
from pathlib import Path

MAX_OUTPUT_BYTES = 2_000_000
MAX_PROBE_SECONDS = 120

# Runs in the container's interpreter, not on the host. Outer Docker execution
# owns the process group/deadline; close() stops the disposable container.
_PROBE_RUNNER = r'''
import base64, json, os, selectors, subprocess, sys, signal
request = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))
argv = [b'/bin/bash', b'-e', b'-o', b'pipefail', b'-c',
        request['command'].encode('utf-8')]
proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
selector = selectors.DefaultSelector()
selector.register(proc.stdout, selectors.EVENT_READ, sys.stdout.buffer)
selector.register(proc.stderr, selectors.EVENT_READ, sys.stderr.buffer)
size, limited = 0, False
try:
    while selector.get_map() and not limited:
        for key, unused in selector.select(timeout=1):
            block = os.read(key.fileobj.fileno(), 65536)
            if not block:
                selector.unregister(key.fileobj)
                continue
            available = request['max_output'] - size
            key.data.write(block[:max(0, available)])
            key.data.flush()
            size += len(block)
            if size > request['max_output']:
                limited = True
                break
    if limited:
        proc.kill()
        sys.stderr.write('\nPROBE_OUTPUT_LIMIT_EXCEEDED\n')
        sys.stderr.flush()
    code = proc.wait(timeout=5)
finally:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    selector.close()
    proc.stdout.close()
    proc.stderr.close()
sys.exit(125 if limited else code)
'''


def run_probe(environment, command, directory, timeout, *, batch=None, source_version=None):
    """Use an explicit plan-scoped session, or own a one-shot session."""
    if batch is not None:
        return batch.run(command, directory, timeout, source_version)
    from agent.execution.probe_batch import ProbeBatch
    owned = ProbeBatch(environment, Path(directory) / "probe-sessions")
    try:
        return owned.run(command, directory, timeout, source_version)
    finally:
        owned.close()
