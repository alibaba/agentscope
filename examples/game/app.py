# -*- coding: utf-8 -*-
from typing import List
import os
import yaml
import datetime

import agentscope

from utils import (
    CheckpointArgs,
    enable_web_ui,
    send_chat_msg,
    send_player_input,
    get_chat_msg,
    get_suggests,
)

import gradio as gr

# from gradio_groupchat import GroupChat
import modelscope_gradio_components as mgr


enable_web_ui()

glb_history_chat = []
MAX_NUM_DISPLAY_MSG = 20


def export_chat_history():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_filename = f"chat_history_{timestamp}.txt"

    with open(export_filename, "w", encoding="utf-8") as file:
        for role, message in glb_history_chat:
            file.write(f"{role}: {message}\n")

    return gr.update(value=export_filename, visible=True)


def get_chat() -> List[List]:
    """Load the chat info from the queue, and put it into the history

    Returns:
        `List[List]`: The parsed history, list of tuple, [(role, msg), ...]

    """
    global glb_history_chat
    line = get_chat_msg()
    if line is not None:
        glb_history_chat += [line]

    return glb_history_chat[-MAX_NUM_DISPLAY_MSG:]


def fn(data: gr.EventData):
    # print(data._data['value'])
    send_player_input(data._data["value"])
    # send_chat_msg("".join(data._data['value']), "你")


if __name__ == "__main__":

    def start_game():
        with open("./config/game_config.yaml", "r", encoding="utf-8") as file:
            GAME_CONFIG = yaml.safe_load(file)
        TONGYI_CONFIG = {
            "type": "tongyi",
            "name": "tongyi_model",
            "model_name": "qwen-max-1201",
            "api_key": os.environ.get("TONGYI_API_KEY"),
        }

        agentscope.init(model_configs=[TONGYI_CONFIG], logger_level="INFO")
        args = CheckpointArgs()
        args.game_config = GAME_CONFIG
        from main import main

        main(args)

    with gr.Blocks() as demo:
        # Users can select the interested exp

        with gr.Row():
            # chatbot = GroupChat(label="Dialog", show_label=False, height=600)
            chatbot = mgr.Chatbot(label="Dialog", show_label=False, height=600)
        with gr.Row():
            with gr.Column():
                user_chat_input = gr.Textbox(
                    label="user_chat_input",
                    placeholder="想说点什么",
                    show_label=False,
                    interactive=True,
                )

            user_chat_bot_suggest = gr.Dataset(
                label="选择一个",
                components=[user_chat_input],
                samples=[],
                visible=True,
            )

            user_chat_bot_suggest.select(
                lambda evt: evt[0],
                inputs=[user_chat_bot_suggest],
                outputs=[user_chat_input],
            )

        with gr.Column():
            send_button = gr.Button(
                value="发送",
            )

        with gr.Accordion("导出选项", open=False):
            with gr.Column():
                export_button = gr.Button("导出完整游戏记录")
                export_output = gr.File(label="下载完整游戏记录", visible=False)

        def send_message(msg):
            send_player_input(msg)
            send_chat_msg(msg, "你")
            return ""

        def update_suggest():
            msg, samples = get_suggests()
            if msg is not None:
                return gr.Dataset(
                    label=msg,
                    samples=samples,
                    visible=True,
                    components=[user_chat_input],
                )
            else:
                return gr.Dataset(
                    label="选择一个",
                    components=[user_chat_input],
                    samples=[],
                    visible=True,
                )

        outputs = [chatbot, user_chat_bot_suggest]
        send_button.click(send_message, user_chat_input, user_chat_input)
        export_button.click(export_chat_history, [], export_output)
        user_chat_input.submit(send_message, user_chat_input, user_chat_input)

        chatbot.custom(fn=fn)

        demo.load(get_chat, inputs=None, outputs=chatbot, every=0.5)
        demo.load(update_suggest, outputs=user_chat_bot_suggest, every=0.5)
        demo.load(start_game)

    demo.queue()
    demo.launch()
