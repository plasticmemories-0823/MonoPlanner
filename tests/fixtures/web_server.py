"""Offline browser fixture, loaded only by a separate test server config."""
import json
import os
import time
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent.task.contracts import WorkRequest
from agent.server import graph as application
import agent.execution.web as retrieval


class OfflineModel:
    def bind_tools(self, tools):
        return self

    def invoke_with_progress(self, messages, on_reasoning):
        for text in ('正在确认本轮要求。', '继续核对可用材料。', '随后给出公开回复。'):
            on_reasoning(text)
            time.sleep(3 if '流式' in messages[-1].text else .3)
        response = self.invoke(messages)
        response.additional_kwargs['reasoning_content'] = '正在确认本轮要求。继续核对可用材料。随后给出公开回复。'
        return response

    def invoke(self, messages):
        last = messages[-1]
        if isinstance(last, ToolMessage):
            result = json.loads(last.content)
            if last.name == 'web_search' and not result.get('error'):
                return self.web('web_fetch', url='https://example.com/fixture')
            if last.name in {'web_search', 'web_fetch'}:
                return self.text('联网工具的离线结果已收到。来源：https://example.com/fixture；没有调用付费 API。')
            if last.name == 'run_existing':
                return self.text('已有任务运行结束。退出码：' + str(result.get('exit_code')) + '；被运行任务 Token 未知。没有调用付费 API。')
            if last.name in {'read_file', 'list_files', 'search_files'}:
                return self.text('查询已完成：' + (result.get('text') or result.get('error') or '目录已列出') + '。没有调用付费 API。')
            content = '实际读取已完成；这里没有调用付费模型。'
            if result.get('outcome') == 'superseded_by_user':
                content = '已按照补充要求停止旧操作。'
            return self.text(content)
        request = last.text
        try:
            request = json.loads(request).get('request', request)
        except (ValueError, AttributeError):
            pass
        if '慢速' in request:
            time.sleep(25)
        if 'SDK消息样本' in request:
            from agent.model.client import ConfiguredModel
            from tests.test_public_model_stream import CallbackModel
            return ConfiguredModel(CallbackModel(tags=['nostream'])).invoke_with_progress(messages, lambda text: None)
        if '操作批准样本' in request:
            return self.web('request_file_operation', operation='create',
                            path=os.environ['MONOPLANER_BROWSER_FILE'], destination='', content='离线批准样本')
        if '联网' in request:
            return self.web('web_search', query='离线界面验证')
        if '运行样本' in request:
            return self.web('run_existing', entry='task.py', args=[], timeout=30, usage_file='')
        if '查询文件' in request:
            return self.web('read_file', path='demo.py', offset=0, limit=80, reread_reason='')
        if '读取' in request:
            value = WorkRequest.model_validate({'kind': 'inspect', 'note': 'private fixture note',
                'steps': [{'id': 'readDemo', 'op': 'read', 'path': 'demo.py'}]}).model_dump()
            return AIMessage(content='我会读取指定的小文件。', tool_calls=[{
                'name': 'submit_plan', 'id': 'offline-read', 'args': value}],
                additional_kwargs={'reasoning_content': 'PRIVATE_BROWSER_REASONING'},
                usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5,
                                'input_token_details': {'cache_read': 2}})
        return self.text('已收到：' + request + '。这是离线浏览器验证，没有调用付费 API。')

    @staticmethod
    def web(name, **args):
        return AIMessage(content='正在查找公开网页。' if name.startswith('web_') else '正在读取指定文件。',
            tool_calls=[{'name': name, 'id': 'offline-'+name, 'args': args}],
            additional_kwargs={'reasoning_content': 'PRIVATE_BROWSER_REASONING'},
            usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5,
                            'input_token_details': {'cache_read': 2}})

    @staticmethod
    def text(content):
        return AIMessage(content=content,
            additional_kwargs={'reasoning_content': 'PRIVATE_BROWSER_REASONING'},
            usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5,
                            'input_token_details': {'cache_read': 2}})


application.model_factory = lambda settings=None: OfflineModel()

# This server also starts the production title worker. Keep its model separate
# from main responses and never allow the offline fixture to read real keys.
from agent.server import titles
from tests.fixtures.title_model import OfflineTitle
titles.model_factory = OfflineTitle


class OfflineWeb:
    def search(self, query, loop, folder):
        return {'sources': [{'title': '离线网页样本', 'url': 'https://example.com/fixture', 'snippet': '固定假搜索结果'}]}

    def fetch(self, url, loop, folder):
        return {'url': url, 'text': '固定假网页正文，没有发出网页 HTTP 请求。'}


retrieval.WebProvider = OfflineWeb
graph = application.graph
