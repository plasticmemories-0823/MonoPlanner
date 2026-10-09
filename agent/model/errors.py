"""模型请求发送前的本地停止信号，不代表远端 API 调用失败。"""


class ModelRequestNotSent(RuntimeError):
    """仅在确认尚未发送请求时抛出；执行器据此撤销预扣的调用额度。"""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


class ModelResponseNotSaved(RuntimeError):
    """API 已返回，但本地记录未能保存；保留已知响应及用量，禁止执行其中工具。"""

    def __init__(self, response, reason):
        super().__init__(reason)
        self.response = response
        self.reason = reason


class ModelCallError(RuntimeError):
    """Redacted transport failure that retains its retry classification."""

    def __init__(self, reason, retryable=False, context_overflow=False):
        super().__init__(reason)
        self.retryable = retryable
        self.context_overflow = context_overflow


def is_context_overflow(exc):
    """Only an explicit provider rejection, never a timeout or arbitrary 400."""
    if isinstance(exc, ModelCallError):
        return exc.context_overflow
    if getattr(exc, 'status_code', None) not in {400, 413}:
        return False
    body = getattr(exc, 'body', None)
    if not isinstance(body, dict):
        return False
    error = body.get('error', body)
    if not isinstance(error, dict):
        return False
    if error.get('code') in {'context_length_exceeded', 'context_window_exceeded', 'CONTEXT_WINDOW_EXCEEDED'}:
        return True
    import re
    text = error.get('message', '')
    return isinstance(text, str) and bool(re.search(
        r'\b(?:maximum|max)(?:\s+(?:allowed|supported))?\s+context\s+(?:length|window)\b'
        r'|\bcontext[\s_-](?:length|window)[\s_-](?:exceeded|overflow)\b'
        r'|\b(?:request|prompt|input|messages?)\s+(?:is\s+|are\s+)?too\s+(?:large|long)\s+for\s+'
        r'(?:(?:this|the)\s+)?(?:model(?:\'s)?\s+)?context(?:\s+window)?\b', text, re.I))


def is_transient_model_error(exc):
    """Use exception types/status codes, never a substring of an error message."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ModelCallError):
            return exc.retryable
        code = getattr(exc, "status_code", None)
        if isinstance(code, int):
            return code in {408, 429} or 500 <= code <= 599
        if isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in {
            "APIConnectionError", "APITimeoutError", "OpenAIConnectionError", "RateLimitError",
            "ConnectError", "ConnectTimeout", "ReadTimeout", "RemoteProtocolError",
        }:
            return True
        exc = exc.__cause__ or exc.__context__
    return False
