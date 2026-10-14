"""为采用生产生命周期的浏览器测试环境提供不联网的标题模型。"""

import time
from langchain_core.messages import AIMessage


class OfflineTitle:
    def invoke(self, messages):
        time.sleep(0.4)
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
