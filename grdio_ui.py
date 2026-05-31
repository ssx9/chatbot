from pathlib import Path
import json

import gradio as gr
from pydantic_ai import ToolCallPart, ToolReturnPart, ThinkingPart
from pydantic import BaseModel
from gradio.components.chatbot import ExampleMessage
from pydantic_ai import Agent


class GradioUI:
    FULLSCREEN_CSS = '''
    html, body {
        margin: 0;
        height: 100%;
    }
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 12px 12px !important;
        height: 100vh !important;
    }
    #chatbot {
        height: calc(100vh - 110px) !important;
    }
    '''

    def __init__(self, agent: Agent, deps):
        self.agent = agent
        self.deps = deps
        self.input_examples = [
            ExampleMessage(text='What is the weather like in New York City?'),
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

    def _process_message_part(self, call, chatbot: list[dict]):
        if isinstance(call, ThinkingPart):
            self._append_thinking_part(call, chatbot)
            return

        if isinstance(call, ToolCallPart):
            self._append_tool_call_part(call, chatbot)
            return

        if isinstance(call, ToolReturnPart):
            self._append_tool_return_part(call, chatbot)

    async def _stream_assistant_text(self, result, chatbot: list[dict]):
        chatbot.append({'role': 'assistant', 'content': ''})
        async for message in result.stream_text():
            chatbot[-1]['content'] = message
            yield self._build_update(chatbot)

    async def stream_from_agent(self, prompt: str, chatbot: list[dict], past_messages: list):
        chatbot.append({'role': 'user', 'content': prompt})
        yield gr.Textbox(interactive=False, value=''), chatbot, gr.skip()
        async with self.agent.run_stream(
                prompt, deps=self.deps, message_history=past_messages
        ) as result:
            for message in result.new_messages():
                for call in message.parts:
                    self._process_message_part(call, chatbot)
                    yield self._build_update(chatbot)

            async for update in self._stream_assistant_text(result, chatbot):
                yield update

            yield gr.Textbox(interactive=True), gr.skip(), result.all_messages()

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

    def build_demo(self):
        with gr.Blocks(fill_height=True) as demo:
            past_messages = gr.State([])
            chatbot = gr.Chatbot(
                label='Helpful Assistant',
                avatar_images=(None, Path('./avator/bot.png')),
                examples=self.input_examples,
                elem_id='chatbot',
            )
            with gr.Row():
                prompt = gr.Textbox(
                    lines=1,
                    show_label=False,
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

        return demo
