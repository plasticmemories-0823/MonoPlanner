"""Exercise LangGraph's real message callback, which plain fake SDKs omit."""
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.graph import END, START, StateGraph, MessagesState

from agent.model.client import ConfiguredModel, create_model
from agent.server.graph import model_factory


class CallbackModel(BaseChatModel):
    @property
    def _llm_type(self):
        return 'offline-message-callback'

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='正在核对入口。'))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(
            content='正在核对入口。', additional_kwargs={'reasoning_content':'本地思考片段'}))
        yield ChatGenerationChunk(message=AIMessageChunk(content='', response_metadata={'finish_reason':'stop'}))


def test_sdk_sees_only_canonical_reply_while_local_reasoning_still_streams():
    for tags in ([], ['nostream']):
        thoughts = []
        model = ConfiguredModel(CallbackModel(tags=tags))
        def reply(state):
            response = model.invoke_with_progress(state['messages'], thoughts.append)
            return {'messages':[AIMessage(content=response.content, id='canonical-reply')]}
        graph = StateGraph(MessagesState)
        graph.add_node('reply',reply).add_edge(START,'reply').add_edge('reply',END)
        events = list(graph.compile().stream({'messages':[HumanMessage(content='检查入口')]}, stream_mode=['messages','values']))
        visible = [event[1][0] for event in events if event[0]=='messages']
        assert thoughts == ['本地思考片段']
        assert any(m.id=='canonical-reply' for m in visible)
        raw = [m for m in visible if m.id!='canonical-reply']
        assert bool(raw) == (not tags)


def test_production_client_suppresses_sdk_raw_messages_without_api_changes():
    with patch('agent.model.deepseek.DeepSeekClient') as client, patch('agent.model.network.model_http_clients',return_value={}):
        create_model({'model_name':'deepseek-flash','api_key':'offline-only'})
    assert client.call_args.kwargs['tags'] == ['nostream']
    assert client.call_args.kwargs['max_retries'] == 0


def test_graph_factory_keeps_nostream_through_real_callback():
    client = CallbackModel(tags=['nostream', 'existing-local-tag'])
    with patch('agent.server.graph.create_model', return_value=ConfiguredModel(client)), \
            patch('agent.server.graph.load_settings', return_value={}):
        model = model_factory()
    assert 'existing-local-tag' in model.client.tags
    thoughts = []
    def reply(state):
        response = model.invoke_with_progress(state['messages'], thoughts.append)
        return {'messages': [AIMessage(content=response.content, id='canonical-reply')]}
    graph = StateGraph(MessagesState)
    graph.add_node('reply', reply).add_edge(START, 'reply').add_edge('reply', END)
    events = list(graph.compile().stream({'messages': [HumanMessage(content='检查入口')]},
                                        stream_mode=['messages', 'values']))
    visible = [event[1][0] for event in events if event[0] == 'messages']
    assert [m.id for m in visible] == ['canonical-reply']
    assert thoughts == ['本地思考片段']
