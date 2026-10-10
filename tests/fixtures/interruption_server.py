"""Offline browser fixture exercising the production cancellable model path."""
import asyncio
from langchain_core.messages import AIMessageChunk
from agent.model.client import ConfiguredModel
from agent.server import graph as application


class OfflineClient:
    def bind_tools(self, tools, strict):
        return self

    async def astream(self, messages, **options):
        text = messages[-1].text
        yield AIMessageChunk(content='', additional_kwargs={'reasoning_content': '正在确认本轮要求。'})
        if '慢速' in text:
            await asyncio.sleep(600)
        yield AIMessageChunk(content='已收到最新消息：'+text+'。这是离线验证，没有调用付费 API。',
            response_metadata={'finish_reason': 'stop'},
            usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5,
                            'input_token_details': {'cache_read': 2}})


application.model_factory = lambda settings: ConfiguredModel(OfflineClient())
graph = application.graph
