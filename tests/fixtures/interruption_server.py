"""离线浏览器测试环境，用于验证生产中的可取消模型调用路径。"""

import asyncio
from langchain_core.messages import AIMessageChunk
from agent.model.client import ConfiguredModel
from agent.server import graph as application


class OfflineClient:
    def bind_tools(self, tools, strict):
        return self

    async def astream(self, messages, **options):
        text = messages[-1].text
        yield AIMessageChunk(
            content='', additional_kwargs={'reasoning_content': '正在确认本轮要求。'}
        )
        if '慢速' in text:
            await asyncio.sleep(600)
        yield AIMessageChunk(
            content='已收到最新消息：' + text + '。这是离线验证，没有调用付费 API。',
            response_metadata={'finish_reason': 'stop'},
            usage_metadata={
                'input_tokens': 3,
                'output_tokens': 2,
                'total_tokens': 5,
                'input_token_details': {'cache_read': 2},
            },
        )


application.model_factory = lambda settings: ConfiguredModel(OfflineClient())
from tests.fixtures.title_model import OfflineTitle
from agent.server import titles

titles.model_factory = OfflineTitle
graph = application.graph
