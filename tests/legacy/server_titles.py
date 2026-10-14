"""受限的后台标题生成，独立于任务和模型循环。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from threading import BoundedSemaphore
import time
import uuid

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    message_to_dict,
    messages_from_dict,
    messages_to_dict,
)

from agent.model.client import ConfiguredModel, ModelPolicy, create_model
from agent.settings import PROJECT_ROOT, load_settings
from agent.storage.preferences import state_root
from tests.legacy.storage_titles import Titles

POLICY = ModelPolicy(max_output_tokens=128, api_timeout=15, reasoning_effort='none')
_dispatcher = None


def model_factory():
    settings = load_settings(
        {'model_name': 'deepseek-flash', 'reasoning_effort': 'none', 'online': False}
    )
    return create_model(settings, POLICY)


def reservation(identity, text):
    text = text[:4000]
    record = {
        'title': ' '.join(text.split())[:80] or '新对话',
        'status': 'pending',
        'revision': 1,
        'input_id': identity,
        'request_id': uuid.uuid4().hex,
        'request_sent': False,
        'created_at': timestamp(),
    }
    try:
        messages = [
            SystemMessage(
                content=(PROJECT_ROOT / 'resources/prompts/title.txt').read_text(encoding='utf-8')
            ),
            HumanMessage(content=text),
        ]
        record['request'] = {
            'model': 'deepseek-flash',
            'messages': messages_to_dict(messages),
            'max_tokens': 128,
            'timeout': 15,
            'max_retries': 0,
            'extra_body': {'thinking': {'type': 'disabled'}},
        }
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, finished_at=timestamp())
    return record


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def close_model(model):
    if not isinstance(model, ConfiguredModel):
        return
    client = getattr(model, 'client', None)
    if client is not None:
        for name in ('http_client', 'http_async_client'):
            transport = getattr(client, name, None)
            if name == 'http_async_client' and transport is not None:
                asyncio.run(transport.aclose())
            elif transport is not None:
                transport.close()


class TitleDispatcher:
    def __init__(self, root, factory=None):
        self.root = root
        self.records = Titles(root)
        self.factory = factory or model_factory
        self.slots = BoundedSemaphore(16)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='conversation-title')
        self.closed = False
        self.records.settle_interrupted()

    def submit(self, thread):
        if self.closed or not self.slots.acquire(blocking=False):
            self.records.update(
                thread,
                {'pending'},
                status='failed',
                error_type='TitleQueueUnavailable',
                finished_at=timestamp(),
            )
            return None
        try:
            future = self.pool.submit(self.generate, thread)
        except Exception:
            self.slots.release()
            raise
        future.add_done_callback(lambda _: self.slots.release())
        return future

    def generate(self, thread):
        model, started = None, None
        try:
            record = self.records.get(thread)
            if not record or record['status'] != 'pending':
                return
            model = self.factory()
            # 先持久化发送标记，再发起网络调用；若此处崩溃，结果仍视为未知，
            # 即使实际尚未发送也不自动重放。
            if not self.records.update(
                thread, {'pending'}, status='running', request_sent=True, started_at=timestamp()
            ):
                return
            started = time.monotonic()
            response = model.invoke(messages_from_dict(record['request']['messages']))
            elapsed = time.monotonic() - started
            text = response.content.strip() if isinstance(response.content, str) else ''
            valid = (
                bool(text)
                and len(text) <= 80
                and '\n' not in text
                and '\r' not in text
                and not response.tool_calls
                and not response.invalid_tool_calls
                and response.response_metadata.get('finish_reason') == 'stop'
            )
            values = {
                'status': 'complete' if valid else 'failed',
                'response': message_to_dict(response),
                'usage': response.usage_metadata,
                'seconds': elapsed,
                'finished_at': timestamp(),
            }
            if valid:
                values['title'] = text
            else:
                values['error_type'] = 'InvalidTitleResponse'
            self.records.update(thread, {'running'}, **values)
        except Exception as exc:
            try:
                self.records.update(
                    thread,
                    {'pending', 'running'},
                    status='unknown' if started else 'failed',
                    error_type=type(exc).__name__,
                    seconds=time.monotonic() - started if started else 0,
                    finished_at=timestamp(),
                )
            except Exception:
                logging.getLogger(__name__).exception('Title record could not be finalized')
        finally:
            try:
                close_model(model)
            except Exception:
                logging.getLogger(__name__).exception('Title model transport could not be closed')

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.records.settle_interrupted()


def schedule(root, thread):
    # CLI 和图单元测试不启动 HTTP 服务生命周期，也不在这些路径创建客户端；
    # 只有本服务新预留的标题任务可以运行。
    try:
        if _dispatcher is not None and _dispatcher.root == root:
            _dispatcher.submit(thread)
        else:
            Titles(root).update(
                thread,
                {'pending'},
                status='failed',
                error_type='TitleServiceUnavailable',
                finished_at=timestamp(),
            )
    except Exception as exc:
        try:
            Titles(root).update(
                thread,
                {'pending'},
                status='failed',
                error_type=type(exc).__name__,
                finished_at=timestamp(),
            )
        except Exception:
            logging.getLogger(__name__).exception('Title scheduling record could not be finalized')


@asynccontextmanager
async def lifespan(app):
    global _dispatcher
    dispatcher = await asyncio.to_thread(lambda: TitleDispatcher(state_root()))
    _dispatcher = dispatcher
    try:
        yield
    finally:
        _dispatcher = None
        await asyncio.to_thread(dispatcher.close)
        from agent.execution.preparation import close_jobs

        await asyncio.to_thread(lambda: close_jobs(state_root()))
