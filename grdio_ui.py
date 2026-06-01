from pathlib import Path
import json

import gradio as gr
from pydantic import BaseModel
from gradio.components.chatbot import ExampleMessage
from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)


class GradioUI:
    FULLSCREEN_CSS = '''
    #chatbot {
        height: calc(100vh - 200px) !important;
    }
    #debug_show {
        height: calc(100vh - 200px) !important;
    }
    '''

    def __init__(self, agent: Agent, deps):
        self.agent = agent
        self.deps = deps
        self.input_examples = [
            ExampleMessage(text='北京的天气怎么样？'),
            ExampleMessage(text='上海天气如何？'),
        ]
        self.demo = self.build_demo()

    @staticmethod
    def _serialize_tool_content(content) -> str:
        if isinstance(content, BaseModel):
            return content.model_dump_json()
        return json.dumps(content)

    @staticmethod
    def _find_tool_message(chatbot: list[dict], tool_call_id: str | None):
        if tool_call_id is None:
            return None

        for gr_message in chatbot:
            metadata = gr_message.get('metadata')
            if metadata is None:
                continue
            tool_id = metadata.get('id', '')
            if tool_id == tool_call_id:
                return gr_message
        return None

    @staticmethod
    def _build_update(chatbot: list[dict]):
        return gr.skip(), chatbot, gr.skip()

    @staticmethod
    def _append_thinking_part(call: ThinkingPart, chatbot: list[dict]):
        chatbot.append(
            {
                'role': 'assistant',
                'content': call.content,
                'metadata': {'title': '🧠️ Thinking:'},
            }
        )

    @staticmethod
    def _append_tool_call_part(call: ToolCallPart, chatbot: list[dict]):
        display_name = call.tool_name
        metadata = {'title': f'🛠️ Using {display_name}'}
        if call.tool_call_id is not None:
            metadata['id'] = call.tool_call_id

        chatbot.append(
            {
                'role': 'assistant',
                'content': 'Parameters: ' + call.args_as_json_str(),
                'metadata': metadata,
            }
        )

    def _append_tool_return_part(self, call: ToolReturnPart, chatbot: list[dict]):
        gr_message = self._find_tool_message(chatbot, call.tool_call_id)
        if gr_message is None:
            return

        json_content = self._serialize_tool_content(call.content)
        gr_message['content'] += f'\nOutput: {json_content}'

    async def stream_from_agent(self, prompt: str, chatbot: list[dict], past_messages: list):
        chatbot.append({'role': 'user', 'content': prompt})
        yield gr.Textbox(interactive=False, value=''), chatbot, gr.skip()

        assistant_message_idx: int | None = None
        thinking_message_idx_by_part: dict[int, int] = {}
        final_messages = past_messages

        def ensure_assistant_message() -> int:
            nonlocal assistant_message_idx
            if assistant_message_idx is None:
                chatbot.append({'role': 'assistant', 'content': ''})
                assistant_message_idx = len(chatbot) - 1
            return assistant_message_idx

        try:
            async with self.agent.run_stream_events(
                    prompt, deps=self.deps, message_history=past_messages
            ) as events:
                async for event in events:
                    if isinstance(event, PartStartEvent):
                        if isinstance(event.part, TextPart):
                            message_idx = ensure_assistant_message()
                            chatbot[message_idx]['content'] += event.part.content
                            yield self._build_update(chatbot)
                        elif isinstance(event.part, ThinkingPart):
                            self._append_thinking_part(event.part, chatbot)
                            thinking_message_idx_by_part[event.index] = len(chatbot) - 1
                            yield self._build_update(chatbot)

                    elif isinstance(event, PartDeltaEvent):
                        if isinstance(event.delta, TextPartDelta):
                            message_idx = ensure_assistant_message()
                            chatbot[message_idx]['content'] += event.delta.content_delta
                            yield self._build_update(chatbot)
                        elif isinstance(event.delta, ThinkingPartDelta):
                            message_idx = thinking_message_idx_by_part.get(event.index)
                            if message_idx is None:
                                chatbot.append(
                                    {
                                        'role': 'assistant',
                                        'content': '',
                                        'metadata': {'title': '🧠️ Thinking:'},
                                    }
                                )
                                message_idx = len(chatbot) - 1
                                thinking_message_idx_by_part[event.index] = message_idx

                            if event.delta.content_delta:
                                chatbot[message_idx]['content'] += event.delta.content_delta
                                yield self._build_update(chatbot)

                    elif isinstance(event, FunctionToolCallEvent):
                        self._append_tool_call_part(event.part, chatbot)
                        yield self._build_update(chatbot)

                    elif isinstance(event, FunctionToolResultEvent):
                        if isinstance(event.part, ToolReturnPart):
                            self._append_tool_return_part(event.part, chatbot)
                            yield self._build_update(chatbot)

                    elif isinstance(event, AgentRunResultEvent):
                        final_messages = event.result.all_messages()
        except Exception as exc:
            error_text = str(exc).strip() or exc.__class__.__name__
            chatbot.append({'role': 'assistant', 'content': f'⚠️ 处理请求时出错：{error_text}'})
            yield self._build_update(chatbot)

        yield gr.Textbox(interactive=True), gr.skip(), final_messages

    async def handle_retry(self, chatbot, past_messages: list, retry_data: gr.RetryData):
        new_history = chatbot[: retry_data.index]
        previous_prompt = chatbot[retry_data.index]['content']
        past_messages = past_messages[: retry_data.index]
        async for update in self.stream_from_agent(previous_prompt, new_history, past_messages):
            yield update

    def undo(self, chatbot, past_messages: list, undo_data: gr.UndoData):
        new_history = chatbot[: undo_data.index]
        past_messages = past_messages[: undo_data.index]
        return chatbot[undo_data.index]['content'], new_history, past_messages

    @staticmethod
    def select_data(message: gr.SelectData) -> str:
        return message.value['text']

    @staticmethod
    def get_user(request: gr.Request):
        return f"当前用户：{request.username}"

    @staticmethod
    def show_messages(messages: list):
        return json.dumps(messages, ensure_ascii=False, indent=2)

    def build_demo(self):
        with gr.Blocks(fill_height=True) as demo:
            past_messages = gr.State([])
            user_info = gr.Markdown()
            demo.load(
                self.get_user,
                outputs=user_info
            )

            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(
                        label='Helpful Assistant',
                        avatar_images=(None, Path('./avator/bot.png')),
                        examples=self.input_examples,
                        elem_id='chatbot',
                    )
                    prompt = gr.Textbox(
                        lines=1,
                        show_label=False,
                    )
                with gr.Column():
                    show_message = gr.Textbox(interactive=False, elem_id='debug_show')
                    debug = gr.Button(
                        value='查看消息记录',
                    )

            prompt.submit(
                self.stream_from_agent,
                inputs=[prompt, chatbot, past_messages],
                outputs=[prompt, chatbot, past_messages],
            )
            chatbot.example_select(self.select_data, None, [prompt])
            chatbot.retry(
                self.handle_retry, [chatbot, past_messages], [prompt, chatbot, past_messages]
            )
            chatbot.undo(self.undo, [chatbot, past_messages], [prompt, chatbot, past_messages])
            debug.click(
                self.show_messages,
                inputs=chatbot,
                outputs=show_message,
            )

        return demo
