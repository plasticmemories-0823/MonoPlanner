"""CLI 与评测运行器共用的 DeepSeek 请求设置。"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelPolicy:
    max_output_tokens: int = 256_000
    api_timeout: float = 90
    reasoning_effort: str = 'high'

    def __post_init__(self):
        for value in (self.max_output_tokens,):
            if type(value) is not int or not 0 < value <= 393_216:
                raise ValueError("模型输出上限必须是 1 到 393216 之间的整数")
        if self.api_timeout <= 0:
            raise ValueError("API timeout 必须大于 0")
        if self.reasoning_effort not in {'none', 'low', 'high', 'max'}:
            raise ValueError('Unsupported reasoning effort')


@dataclass(frozen=True)
class ConfiguredModel:
    """将服务商选项隔离在 Agent 循环之外；此处不创建独立 Agent。"""

    client: object
    policy: ModelPolicy = ModelPolicy()
    tools: tuple = ()
    purpose: str = "work"
    timeout: float | None = None

    @property
    def max_output_tokens(self):
        if self.purpose == 'rag_rewrite':
            return min(self.policy.max_output_tokens, 1024)
        if self.purpose == "compaction":
            from agent.model.compaction import CONFIG

            return min(self.policy.max_output_tokens, CONFIG["summary_max_tokens"])
        return self.policy.max_output_tokens

    @property
    def request_options(self):
        options = {
            "max_tokens": self.max_output_tokens,
            "extra_body": {
                "thinking": {
                    "type": "disabled" if self.policy.reasoning_effort == 'none' else "enabled"
                }
            },
            "timeout": self.timeout if self.timeout is not None else self.policy.api_timeout,
        }
        if self.policy.reasoning_effort != 'none':
            options['reasoning_effort'] = self.policy.reasoning_effort
        if self.purpose in {"compaction", "rag_rewrite"}:
            options["tool_choice"] = "none"
        return options

    def bind_tools(self, tools):
        return replace(self, tools=tuple(tools))

    @property
    def context_envelope(self):
        model_name = getattr(self.client, "model_name", None)
        return {
            "model": model_name if isinstance(model_name, str) else type(self.client).__name__,
            **{key: value for key, value in self.request_options.items() if key != "timeout"},
        }

    def for_purpose(self, purpose, timeout=None):
        return replace(self, purpose=purpose, timeout=timeout)

    def invoke(self, messages):
        from agent.task.control import current

        # 是否可取消由任务控制决定，与界面是否展示思考无关。
        if current() is not None:
            return self.invoke_with_progress(messages)
        bound = self.client.bind_tools(list(self.tools), strict=True) if self.tools else self.client
        return bound.invoke(messages, **self.request_options)

    def invoke_with_progress(self, messages, on_reasoning=None):
        """以流方式处理单次请求；完成前仅公开思考内容。"""
        from langchain_core.messages import message_chunk_to_message
        from agent.model.payload import retain_provider_calls

        from agent.task.control import current

        signal = current()
        if signal:
            signal.check()
        bound = self.client.bind_tools(list(self.tools), strict=True) if self.tools else self.client
        if signal:
            from agent.model.errors import ModelInterrupted
            from agent.model.streaming import run

            dispatched = False

            async def streamed():
                import asyncio

                nonlocal dispatched
                # 从同步线程排入异步循环后，暂停可能已到达。
                signal.check()
                complete = None
                stream = bound.astream(messages, stream_usage=True, **self.request_options)
                dispatched = True
                try:
                    async for chunk in stream:
                        thought = chunk.additional_kwargs.get('reasoning_content')
                        if on_reasoning is not None and isinstance(thought, str) and thought:
                            # 本地展示记录在网络循环之外写入。
                            # 确认取消前，先完成正在进行的写入。
                            update = asyncio.create_task(asyncio.to_thread(on_reasoning, thought))
                            try:
                                await asyncio.shield(update)
                            except asyncio.CancelledError:
                                await update
                                raise
                        complete = chunk if complete is None else complete + chunk
                finally:
                    await stream.aclose()
                return complete

            try:
                complete = run(streamed(), signal)
            except ModelInterrupted as exc:
                # 协程尚未启动就被取消时，SDK 没有收到请求。
                exc.sent = dispatched
                raise
        else:
            complete = None
            for chunk in bound.stream(messages, stream_usage=True, **self.request_options):
                thought = chunk.additional_kwargs.get('reasoning_content')
                if on_reasoning is not None and isinstance(thought, str) and thought:
                    on_reasoning(thought)
                complete = chunk if complete is None else complete + chunk
        if complete is None or not complete.response_metadata.get('finish_reason'):
            raise RuntimeError('Model stream ended without a complete response')
        response = message_chunk_to_message(complete)
        if complete.tool_call_chunks:
            retain_provider_calls(
                response,
                [
                    {
                        'id': call['id'],
                        'type': 'function',
                        'function': {'name': call['name'], 'arguments': call['args']},
                    }
                    for call in complete.tool_call_chunks
                ],
            )
        return response


def configure_call(model, purpose, timeout=None):
    # 检查类本身，避免使用测试替身动态生成的属性。
    if callable(getattr(type(model), "for_purpose", None)):
        return model.for_purpose(purpose, timeout=timeout)
    return model


def request_timeout(model, remaining=None):
    """遵循显式调用配置和模型策略，并受任务剩余时间约束。"""
    from agent.model.errors import ModelRequestNotSent

    policy = getattr(model, 'policy', None)
    configured = getattr(model, 'timeout', None)
    if not isinstance(configured, (int, float)) or isinstance(configured, bool):
        configured = (
            policy.api_timeout if isinstance(policy, ModelPolicy) else ModelPolicy().api_timeout
        )
    if remaining is not None:
        if remaining <= 0:
            raise ModelRequestNotSent('task_timeout', 'Task time limit reached before model dispatch.')
        configured = min(configured, remaining)
    return configured


def create_model(settings, policy=None):
    """加载本模块时不读取凭据，也不创建 API 客户端。"""
    from agent.model.deepseek import DeepSeekClient
    from agent.model.network import model_http_clients

    policy = policy or ModelPolicy(reasoning_effort=settings.get('reasoning_effort', 'high'))
    client = DeepSeekClient(
        model=settings["model_name"],
        api_key=settings["api_key"],
        base_url="https://api.deepseek.com/beta",
        temperature=settings.get("temperature", 0),
        max_tokens=policy.max_output_tokens,
        timeout=policy.api_timeout,
        max_retries=0,
        tags=['nostream'],
        **model_http_clients(policy.api_timeout),
    )
    return ConfiguredModel(client, policy)
