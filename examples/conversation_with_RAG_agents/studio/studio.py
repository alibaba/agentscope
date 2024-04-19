# -*- coding: utf-8 -*-
"""run web ui"""
import argparse
import os
import sys
import threading
import time
import json
from collections import defaultdict
from typing import Optional, Callable
import traceback

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    import modelscope_studio as mgr
except ImportError:
    mgr = None

from agentscope.web.studio.utils import (
    send_player_input,
    get_chat_msg,
    SYS_MSG_PREFIX,
    ResetException,
    check_uuid,
    send_msg,
    generate_image_from_name,
    send_reset_msg,
    thread_local_data,
)

MAX_NUM_DISPLAY_MSG = 20
FAIL_COUNT_DOWN = 30


def init_uid_list() -> list:
    """Initialize an empty list for storing user IDs."""
    return []


glb_history_dict = defaultdict(init_uid_list)
glb_signed_user = []


def get_new_uid(request: gr.Request) -> str:
    """extract user ID from request"""
    headers = request.headers
    splits = headers.get("referer").split("?", 1)
    if len(splits) > 1:
        uid = splits[1]
    else:
        uid = "local_user"
    return uid


def reset_glb_var(uid: str) -> None:
    """Reset global variables for a given user ID."""
    global glb_history_dict
    glb_history_dict[uid] = init_uid_list()


def get_chat(uid: str) -> list[list]:
    """Retrieve chat messages for a given user ID."""
    uid = check_uuid(uid)
    global glb_history_dict
    line = get_chat_msg(uid=uid)
    # TODO: Optimize the display effect, currently there is a problem of
    #  output display jumping
    if line:
        glb_history_dict[uid] += [line]
    dial_msg = []
    for line in glb_history_dict[uid]:
        _, msg = line
        if isinstance(msg, dict):
            dial_msg.append(line)
        else:
            # User chat, format: (msg, None)
            dial_msg.append(line)
    return dial_msg[-MAX_NUM_DISPLAY_MSG:]


def send_image(image_term: str, uid: str) -> None:
    """Send an image as a chat message."""
    uid = check_uuid(uid)
    send_player_input(image_term, uid=uid)

    msg = f"""<img src="{image_term}"></img>"""
    avatar = generate_image_from_name("Me")
    send_msg(msg, is_player=True, role="Me", uid=uid, avatar=avatar)


def send_message(msg: str, uid: str) -> str:
    """Send a generic message to the player."""
    uid = check_uuid(uid)
    send_player_input(msg, uid=uid)
    avatar = generate_image_from_name("Me")
    send_msg(msg, is_player=True, role="Me", uid=uid, avatar=avatar)
    return ""


def fn_choice(data: gr.EventData, uid: str) -> None:
    """Handle a selection event from the chatbot interface."""
    uid = check_uuid(uid)
    # pylint: disable=protected-access
    send_player_input(data._data["value"], uid=uid)


def import_function_from_path(
    module_path: str,
    function_name: str,
    module_name: Optional[str] = None,
) -> Callable:
    """Import a function from the given module path."""
    import importlib.util

    script_dir = os.path.dirname(os.path.abspath(module_path))

    # Temporarily add a script directory to sys.path
    original_sys_path = sys.path[:]
    sys.path.insert(0, script_dir)

    try:
        # If a module name is not provided, you can use the filename (
        # without extension) as the module name
        if module_name is None:
            module_name = os.path.splitext(os.path.basename(module_path))[0]
        # Creating module specifications and loading modules
        spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
        if spec is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Getting a function from a module
            function = getattr(module, function_name)
        else:
            raise ImportError(
                f"Could not find module spec for {module_name} at"
                f" {module_path}",
            )
    except AttributeError as exc:
        raise AttributeError(
            f"The module '{module_name}' does not have a function named '"
            f"{function_name}'. Please put your code in the main function, "
            f"read README.md for details.",
        ) from exc
    finally:
        # Restore the original sys.path
        sys.path = original_sys_path

    return function


# pylint: disable=too-many-statements
def run_app() -> None:
    """Entry point for the web UI application."""
    assert gr is not None, "Please install [full] version of AgentScope."

    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str, help="Script file to run")
    args = parser.parse_args()

    # Make sure script_path is an absolute path
    script_path = os.path.abspath(args.script)

    # Get the directory where the script is located
    script_dir = os.path.dirname(script_path)
    # Save the current working directory
    # Change the current working directory to the directory where
    os.chdir(script_dir)

    def start_game(uid: str) -> None:
        """Start the main game loop."""
        thread_local_data.uid = uid
        main = import_function_from_path(script_path, "main")

        while True:
            try:
                main()
            except ResetException:
                print(f"Reset Successfully：{uid} ")
            except Exception as e:
                trace_info = "".join(
                    traceback.TracebackException.from_exception(e).format(),
                )
                for i in range(FAIL_COUNT_DOWN, 0, -1):
                    send_msg(
                        f"{SYS_MSG_PREFIX} error {trace_info}, reboot "
                        f"in {i} seconds",
                        uid=uid,
                    )
                    time.sleep(1)
            reset_glb_var(uid)

    def check_for_new_session(uid: str) -> None:
        """
        Check for a new user session and start a game thread if necessary.
        """
        uid = check_uuid(uid)
        if uid not in glb_signed_user:
            glb_signed_user.append(uid)
            print("==========Signed User==========")
            print(f"Total number of users: {len(glb_signed_user)}")
            run_thread = threading.Thread(
                target=start_game,
                args=(uid,),
            )
            run_thread.start()

    with gr.Blocks(css="studio/assets/app.css") as demo:
        warning_html_code = """
                        <div class="hint" style="text-align:
                        center;background-color: rgba(255, 255, 0, 0.15);
                        padding: 10px; margin: 10px; border-radius: 5px;
                        border: 1px solid #ffcc00;">
                        <p>If you want to start over, please click the
                        <strong>reset</strong>
                        button and <strong>refresh</strong> the page</p>
                        </div>
                        """
        gr.HTML(warning_html_code)
        uuid = gr.Textbox(label="modelscope_uuid", visible=False)
        tabs = gr.Tabs(visible=True)
        with tabs:
            conversation_tab = gr.Tab("Q&A", id=0)
            agent_tab = gr.Tab("available agents / 可选智能体", id=1)
            with conversation_tab:
                with gr.Row():
                    chatbot = mgr.Chatbot(
                        label="Dialog",
                        show_label=False,
                        bubble_full_width=False,
                        visible=True,
                    )
        agent_buttons = []
        with agent_tab:
            agent_info = """
            <div class="agent-box">
                <p>
                    According to the information in agent_config.json,
                    there are following agents ready to answer
                    questions for you.
                    <br>
                    根据agent_config.json中的配置信息，有以下agent为您答疑解惑。
                </p>
            </div>
            """
            gr.HTML(agent_info)
            with open("configs/agent_config.json", "r", encoding="utf-8") as f:
                agent_configs = json.load(f)
            for config in agent_configs:
                agent_args = config["args"]
                button_value = (
                    f'{agent_args["name"]}: '
                    f'\n 🔥{agent_args.get("description", "No description.")}'
                )
                with gr.Row():
                    agent_buttons.append(
                        (
                            gr.Button(
                                value=button_value,
                                elem_classes=["button"],
                            ),
                            gr.Textbox(
                                value=agent_args["name"],
                                visible=False,
                            ),
                        ),
                    )

        with gr.Column():
            user_chat_input = gr.Textbox(
                label="user_chat_input",
                placeholder="Say something here",
                show_label=False,
                interactive=True,
            )
            send_button = gr.Button(value="📣Send")
        with gr.Column():
            reset_button = gr.Button(value="Reset")

        # submit message
        send_button.click(
            send_message,
            [user_chat_input, uuid],
            user_chat_input,
        )
        user_chat_input.submit(
            send_message,
            [user_chat_input, uuid],
            user_chat_input,
        )

        def mention(text: str) -> str:
            return f"@{text}"

        for button_pair in agent_buttons:
            button_pair[0].click(
                fn=mention,
                inputs=[button_pair[1]],
                outputs=user_chat_input,
            )

        reset_button.click(send_reset_msg, inputs=[uuid])

        demo.load(get_new_uid, None, uuid).then(
            check_for_new_session,
            inputs=[uuid],
        ).then(
            get_chat,
            inputs=[uuid],
            outputs=[chatbot],
            every=0.5,
        )

        chatbot.custom(fn=fn_choice, inputs=[uuid])
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    run_app()
