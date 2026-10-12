"""No-network title model for browser fixtures with production lifespans."""
import time
from langchain_core.messages import AIMessage


class OfflineTitle:
    def invoke(self, messages):
        time.sleep(.4)
        return AIMessage(content='离线会话标题验证', response_metadata={'finish_reason': 'stop'},
                         usage_metadata={'input_tokens': 8, 'output_tokens': 4, 'total_tokens': 12,
                                         'input_token_details': {'cache_read': 2}})
