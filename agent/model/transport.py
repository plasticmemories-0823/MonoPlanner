"""关闭本次调用拥有的网络连接，不操作任何请求或调度队列。"""

import asyncio

from agent.model.client import ConfiguredModel


def close_model(model):
    if not isinstance(model, ConfiguredModel):
        return
    client = getattr(model, 'client', None)
    if client is None:
        return
    for name in ('http_client', 'http_async_client'):
        transport = getattr(client, name, None)
        if transport is None:
            continue
        if name == 'http_async_client':
            asyncio.run(transport.aclose())
        else:
            transport.close()
