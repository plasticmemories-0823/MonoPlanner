"""Provider-reported parent usage, never inferred from subprocess output."""


def summarize(state):
    rows = [*state.get('model_usage', []), *state.get('web_usage', [])]
    total = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cached_input_tokens': 0,
             'unknown_calls': 0, 'calls': state.get('budget', {}).get('used_steps', len(rows)), 'model_seconds': 0.0,
             'elapsed_seconds': state.get('elapsed_seconds', 0), 'source': 'assistant_provider_responses'}
    for row in rows:
        total['model_seconds'] += row.get('seconds', 0)
        usage = row.get('usage')
        if not usage or not all(type(usage.get(k)) is int for k in ('input_tokens', 'output_tokens', 'total_tokens')):
            total['unknown_calls'] += 1
            continue
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            total[key] += usage[key]
        total['cached_input_tokens'] += usage.get('input_token_details', {}).get('cache_read', 0)
    total['uncached_input_tokens'] = total['input_tokens'] - total['cached_input_tokens']
    total['cache_rate'] = total['cached_input_tokens'] / total['input_tokens'] if total['input_tokens'] else None
    total['unknown_calls'] = max(total['unknown_calls'], state.get('budget', {}).get('unknown_usage_steps', 0))
    return total
