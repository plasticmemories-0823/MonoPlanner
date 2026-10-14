"""使用真实生产图和模型工厂，提供密集的离线 SDK 流。"""

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from agent.model.client import ConfiguredModel
from tests.legacy import graph as application
from agent.server.app import app
from tests.legacy import server_titles as titles
from tests.fixtures.title_model import OfflineTitle
from tests.test_public_model_stream import CallbackModel


class BurstModel(CallbackModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for index in range(2048):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content='', additional_kwargs={'reasoning_content': f'思考片段{index}。'}
                )
            )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content='密集流式回复已完成。',
                usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15},
            )
        )
        yield ChatGenerationChunk(
            message=AIMessageChunk(content='', response_metadata={'finish_reason': 'stop'})
        )


application.create_model = lambda settings: ConfiguredModel(BurstModel(tags=['nostream']))
application.load_settings = lambda settings=None: settings or {}
titles.model_factory = OfflineTitle
graph = application.builder.compile()
