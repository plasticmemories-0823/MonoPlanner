"""仅使用服务商报告的父级用量，不从子进程输出推断。"""


def summarize(state):
    rows = [*state.get('model_usage', []), *state.get('web_usage', [])]
    services = [row for row in rows
                if row.get('kind') == 'service' and row.get('status') != 'not_sent']
    rows = [row for row in rows if row.get('kind') != 'service']
    pending = state.get('web_request_pending') or {}
    if (pending.get('kind') == 'service'
            and not any(row.get('call') == pending.get('call') for row in services)):
        services.append({**pending, 'status': 'unknown', 'cost_micros': None})
    total = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'cached_input_tokens': 0,
        'unknown_calls': 0,
        'calls': state.get('budget', {}).get('used_steps', len(rows)),
        'model_seconds': 0.0,
        'elapsed_seconds': state.get('elapsed_seconds', 0),
        'source': 'assistant_provider_responses',
        'cache_known_input_tokens': 0,
        'cache_unknown_calls': 0,
        'context': state.get('context_usage'),
        'service_calls': len(services),
        'service_cost_micros': sum(row.get('cost_micros') or 0 for row in services),
        'service_reserved_cost_micros': sum(
            row.get('reserved_cost_micros', 0) for row in services),
        'service_unknown_calls': sum(row.get('cost_micros') is None for row in services),
        'service_seconds': sum(row.get('seconds', 0) for row in services),
    }
    for row in rows:
        total['model_seconds'] += row.get('seconds', 0)
        usage = row.get('usage')
        if not usage or not all(
            type(usage.get(k)) is int for k in ('input_tokens', 'output_tokens', 'total_tokens')
        ):
            total['unknown_calls'] += 1
            total['cache_unknown_calls'] += 1
            continue
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            total[key] += usage[key]
        cached = usage.get('input_token_details', {}).get('cache_read')
        if type(cached) is int and 0 <= cached <= usage['input_tokens']:
            total['cached_input_tokens'] += cached
            total['cache_known_input_tokens'] += usage['input_tokens']
        else:
            total['cache_unknown_calls'] += 1
    total['uncached_input_tokens'] = total['input_tokens'] - total['cached_input_tokens']
    known = total['cache_known_input_tokens']
    total['cache_rate'] = total['cached_input_tokens'] / known if known else None
    total['unknown_calls'] = max(
        total['unknown_calls'], state.get('budget', {}).get('unknown_usage_steps', 0)
    )
    total['cache_unknown_calls'] = max(total['cache_unknown_calls'], total['unknown_calls'])
    return total
