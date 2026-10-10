"""Shared DeepSeek request settings for the CLI and evaluation runner."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelPolicy:
    max_output_tokens: int = 65_536
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
    """Keep vendor options out of the Agent loop; no independent Agent here."""

    client: object
    policy: ModelPolicy = ModelPolicy()
    tools: tuple = ()
    purpose: str = "work"
    timeout: float | None = None

    @property
    def max_output_tokens(self):
        if self.purpose == "compaction":
            from agent.model.compaction import CONFIG
            return min(self.policy.max_output_tokens, CONFIG["summary_max_tokens"])
        return self.policy.max_output_tokens

    @property
    def request_options(self):
        options = {"max_tokens": self.max_output_tokens,
                "extra_body": {"thinking": {"type": "disabled" if self.policy.reasoning_effort == 'none' else "enabled"}},
                "timeout": self.timeout if self.timeout is not None else self.policy.api_timeout}
        if self.policy.reasoning_effort != 'none':
            options['reasoning_effort'] = self.policy.reasoning_effort
        if self.purpose == "compaction":
            options["tool_choice"] = "none"
        return options

    def bind_tools(self, tools):
        return replace(self, tools=tuple(tools))

    @property
    def context_envelope(self):
        model_name = getattr(self.client, "model_name", None)
        return {"model": model_name if isinstance(model_name, str) else type(self.client).__name__,
                **{key: value for key, value in self.request_options.items() if key != "timeout"}}

    def for_purpose(self, purpose, timeout=None):
        return replace(self, purpose=purpose, timeout=timeout)

    def invoke(self, messages):
        bound = self.client.bind_tools(list(self.tools), strict=True) if self.tools else self.client
        return bound.invoke(messages, **self.request_options)

    def invoke_with_progress(self, messages, on_reasoning):
        """Stream one request; only reasoning is exposed before completion."""
        from langchain_core.messages import message_chunk_to_message
        from agent.model.payload import RAW_TOOL_CALLS
        bound = self.client.bind_tools(list(self.tools), strict=True) if self.tools else self.client
        from agent.server.interrupts import current
        signal = current()
        if signal:
            from agent.model.errors import ModelInterrupted
            from agent.model.streaming import run
            if signal.kind:
                raise ModelInterrupted(signal.kind, sent=False)

            async def streamed():
                import asyncio
                complete = None
                stream = bound.astream(messages, stream_usage=True, **self.request_options)
                try:
                    async for chunk in stream:
                        thought = chunk.additional_kwargs.get('reasoning_content')
                        if isinstance(thought, str) and thought:
                            # Local display writes run outside the network loop.
                            # Finish an in-flight write before acknowledging cancel.
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

            complete = run(streamed(), signal)
        else:
            complete = None
            for chunk in bound.stream(messages, stream_usage=True, **self.request_options):
                thought = chunk.additional_kwargs.get('reasoning_content')
                if isinstance(thought, str) and thought:
                    on_reasoning(thought)
                complete = chunk if complete is None else complete + chunk
        if complete is None or not complete.response_metadata.get('finish_reason'):
            raise RuntimeError('Model stream ended without a complete response')
        response = message_chunk_to_message(complete)
        if complete.tool_call_chunks:
            response.additional_kwargs[RAW_TOOL_CALLS] = [
                {'id': call['id'], 'type': 'function', 'function': {
                    'name': call['name'], 'arguments': call['args']}}
                for call in complete.tool_call_chunks]
        return response


def configure_call(model, purpose, timeout=None):
    # Check the class, not dynamic attributes produced by test mocks.
    if callable(getattr(type(model), "for_purpose", None)):
        return model.for_purpose(purpose, timeout=timeout)
    return model


def create_model(settings, policy=None):
    """Loading this module does not read credentials or create an API client."""
    from agent.model.deepseek import DeepSeekClient
    from agent.model.network import model_http_clients

    policy = policy or ModelPolicy(reasoning_effort=settings.get('reasoning_effort', 'high'))
    client = DeepSeekClient(model=settings["model_name"], api_key=settings["api_key"],
                          base_url="https://api.deepseek.com/beta", temperature=settings.get("temperature", 0),
                          max_tokens=policy.max_output_tokens, timeout=policy.api_timeout,
                          max_retries=0, **model_http_clients(policy.api_timeout))
    return ConfiguredModel(client, policy)
