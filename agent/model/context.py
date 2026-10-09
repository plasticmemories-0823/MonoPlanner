"""Guard the existing task transcript; do not create a second model history."""
import hashlib
import json
from agent.model.payload import wire_input, estimate_input_tokens
from agent.model.client import ModelPolicy
from agent.model.errors import ModelRequestNotSent
from agent.settings import PROJECT_ROOT


# Installed dsh's Flash catalog uses this combined input/output token window.
CONTEXT_WINDOW_TOKENS = json.loads((PROJECT_ROOT / 'resources/context.json').read_text(encoding='utf-8'))['context_window_tokens']


def fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def admit(state, messages, tools, options):
    payload = wire_input(messages, tools)
    hashes = [fingerprint(m) for m in payload['messages']]
    envelope = fingerprint({'tools': payload.get('tools'), 'options': {
        k: v for k, v in options.items() if k != 'timeout'}})
    old = state.get('last_admission')
    generation = state.get('context_view', {}).get('generation', 0)
    if old:
        rollover = state.get('context_rollover', {})
        changed_generation = generation != old.get('generation', 0)
        authorized = (changed_generation and rollover.get('generation') == generation
                      and rollover.get('previous') == fingerprint(old))
        if (old['envelope'] != envelope or (changed_generation and not authorized)
                or (not changed_generation and hashes[:len(old['messages'])] != old['messages'])):
            raise ModelRequestNotSent('context_prefix_changed', 'Previous input changed; no request sent.')
    input_tokens = estimate_input_tokens(messages, tools)
    output_tokens = options.get('max_tokens', ModelPolicy().max_output_tokens)
    if input_tokens + output_tokens > CONTEXT_WINDOW_TOKENS:
        raise ModelRequestNotSent('context_capacity_exhausted',
            f'Context window exceeded: estimated input {input_tokens} + reserved output '
            f'{output_tokens} > {CONTEXT_WINDOW_TOKENS} tokens; no history was removed.')
    state['last_admission'] = {'messages': hashes, 'envelope': envelope, 'generation': generation}
    state.pop('context_rollover', None)
