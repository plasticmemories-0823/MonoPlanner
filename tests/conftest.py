"""Default offline service isolation; individual tests can override both seams."""
import os

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def offline_conversation_service(tmp_path, monkeypatch):
    monkeypatch.setenv('MONOPLANER_SERVICE_TOKEN', 'offline-instance-token-not-a-production-credential')
    if 'MONOPLANER_STATE_DIR' not in os.environ:
        monkeypatch.setenv('MONOPLANER_STATE_DIR', str(tmp_path / 'conversation-state'))
    from agent.server import titles

    class OfflineTitleModel:
        def invoke(self, messages):
            return AIMessage(content='离线会话标题验证', response_metadata={'finish_reason': 'stop'},
                             usage_metadata={'input_tokens': 8, 'output_tokens': 4, 'total_tokens': 12,
                                             'input_token_details': {'cache_read': 2}})

    monkeypatch.setattr(titles, 'model_factory', OfflineTitleModel)
