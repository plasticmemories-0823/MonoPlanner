"""默认隔离离线服务；各测试可分别替换这两处边界。"""

import os

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def offline_conversation_service(tmp_path, monkeypatch):
    monkeypatch.setenv(
        'MONOPLANER_SERVICE_TOKEN', 'offline-instance-token-not-a-production-credential'
    )
    if 'MONOPLANER_STATE_DIR' not in os.environ:
        monkeypatch.setenv('MONOPLANER_STATE_DIR', str(tmp_path / 'conversation-state'))
    from tests.legacy import server_titles as titles

    class OfflineTitleModel:
        def invoke(self, messages):
            return AIMessage(
                content='离线会话标题验证',
                response_metadata={'finish_reason': 'stop'},
                usage_metadata={
                    'input_tokens': 8,
                    'output_tokens': 4,
                    'total_tokens': 12,
                    'input_token_details': {'cache_read': 2},
                },
            )

    monkeypatch.setattr(titles, 'model_factory', OfflineTitleModel)
    from agent.application import titles as application_titles

    monkeypatch.setattr(application_titles, 'model_factory', OfflineTitleModel)
    # 环境缺失的对话测试绝不能下载生产镜像。
    # 托管环境准备测试会明确替换此处边界。
    from agent.execution import preparation
    from agent.execution.projects import EnvironmentRequired

    def offline_prepare(*args, **kwargs):
        raise EnvironmentRequired('Docker 离线测试未准备环境；不会下载依赖。')

    monkeypatch.setattr(preparation, 'ensure_prepared', offline_prepare)
    yield
    # 无 lifespan 的读取测试也可能懒创建服务；不能让它跨测试引用前一个临时目录。
    from agent.application import api

    if api._service is not None:
        api._service.close()
        api._service = None
