# -*- coding: utf-8 -*-
# flake8: noqa
# pylint: skip-file
"""
This is a Gradio interface that can connect to the server
"""
import os
import traceback
from http import HTTPStatus
import dashscope
import requests
from loguru import logger
from typing import cast, Any

import gradio as gr
from gradio.components.chatbot import FileDataDict, MessageDict
import modelscope_studio as mgr
from modelscope_studio.components.Chatbot import MultimodalMessage
from modelscope_studio.components.MultimodalInput import MultimodalInputData

service_url = os.getenv("MODEL_SERVICE_URL", "http://localhost:5006/api")
feedback_service_url = os.getenv(
    "FEEDBACK_SERVICE_URL",
    "http://localhost:5006/api/feedback",
)

answer_input_js = """(props, cc, { el, onMount, onUpdate }) => {
      el.innerHTML = ` <style>
      .scaled-content {
            transform: scale(0.8);
            transform-origin: top left;
      }
      .button-style {
        transition: transform 0.1s;
      }
      .button-style:active {
        transform: scale(0.95);
      }
      </style>
        <br><br>
        <div class="scaled-content">
        <select name="scores" id="scores">
        <option value=1>1分</option>
        <option value=0.5>0.5分</option>
        <option value=0>0分</option>
      </select>
      <button class="button-style" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">提交分数</button>
      <details>
      <summary>输入建议/更正后答案</summary>
      <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px;'>
        <input type='text' id='suggest-answer' style='margin-top: 5px; width: 100%;'>
        <button class="button-style" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">提交</button>
</div>
</details>
<\div>`
    onUpdate(() => {
        el.getElementsByTagName('button')[1].addEventListener('click', () => {
            var suggest_answer = el.getElementsByTagName('input')[0].value;
            console.log('suggest answer:', suggest_answer);
            console.log(props);
            cc.dispatch(JSON.stringify({
                    suggest_answer: suggest_answer, // Data to be sent
                    id: props['id']
            }))
            el.getElementsByTagName('button')[1].innerHTML = "已提交"
        });
        el.getElementsByTagName('button')[0].addEventListener('click', () => {
            var score = el.getElementsByTagName('select')[0].value;
            console.log('score:', document.getElementsByTagName('select'));
            console.log('score:', score);
            console.log(props);
            cc.dispatch(JSON.stringify({
                    score: score, // Data to be sent
                    id: props['id']
            }))
            el.getElementsByTagName('button')[0].innerHTML = "已提交"
        });
    } ,  { callAfterMount: true })
}
"""


def insert_request_id(request_id: str) -> str:
    return f"""<answer-input id='{request_id}'></answer-input>"""


def _extract_from_img(msg: MultimodalInputData) -> str:
    if isinstance(msg, dict):
        msg = MultimodalInputData(**msg)
    if msg.files is None or len(msg.files) == 0:
        return ""

    messages = [
        {
            "role": "user",
            "content": [{"image": "file://" + f.path} for f in msg.files]
            + [{"text": "你是一个图片信息提取助手。图片中包含了什么信息?请尽可能详细地描述。"}],
        },
    ]
    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    response = dashscope.MultiModalConversation.call(
        model="qwen-vl-max",
        api_key=os.getenv("DASHSCOPE_API_KEY", "secret-api-key-1"),
        messages=messages,
        result_format="messages",  # 将返回结果格式设置为 message
    )
    return response["output"]["choices"][0]["message"].content[0]["text"]


def chat(message: Any, history: Any) -> Any:
    """chat function in gradio"""
    history_dashscope_format = []
    for i, (usr_msg, bot_msg) in enumerate(history):
        if isinstance(usr_msg, dict):
            history[i][0] = MultimodalMessage(**usr_msg)
        if isinstance(bot_msg, dict):
            history[i][1] = MultimodalMessage(**bot_msg)

        if isinstance(usr_msg, list):
            usr_msg = "".join(m.text for m in usr_msg)
        if isinstance(bot_msg, list):
            bot_msg = "".join(m.text for m in bot_msg)

        if isinstance(usr_msg, MultimodalMessage):
            usr_msg = usr_msg.text
        if isinstance(bot_msg, MultimodalMessage):
            bot_msg = bot_msg.text
        history_dashscope_format.append(
            {
                "role": "user",
                "content": usr_msg if usr_msg is not None else "",
            },
        )
        history_dashscope_format.append(
            {
                "role": "assistant",
                "content": bot_msg if bot_msg is not None else "",
            },
        )

    if isinstance(message, MultimodalInputData) or isinstance(message, dict):
        extracted_info = _extract_from_img(message)
        usr_msg = message.get("text", "")
        if len(extracted_info) > 0:
            usr_msg += "\n用户上传图片中的文字信息:\n" + extracted_info
            yield "\n用户上传图片中的文字信息:" + extracted_info
    else:
        usr_msg = message

    history_dashscope_format.append(
        {"role": "user", "content": usr_msg},
    )

    print(history_dashscope_format)
    dashscope.base_http_api_url = service_url
    is_stream = True
    responses = dashscope.Generation.call(
        model=os.getenv("TEST_MODEL", "test"),
        api_key=os.getenv("DASHSCOPE_API_KEY", "secret-api-key-1"),
        messages=history_dashscope_format,
        result_format="messages",  # 将返回结果格式设置为 message
        stream=is_stream,
    )
    full_content = ""
    if is_stream:
        try:
            for response in responses:
                request_id = response.get("request_id", "")
                if response.status_code == HTTPStatus.OK:
                    # logger.info(str(response))
                    try:
                        full_content = response.output.choices[0].messages[0][
                            "content"
                        ]
                    except:
                        pass
                    yield full_content + insert_request_id(request_id)
                else:
                    print(
                        "Request id: %s, Status code: %s, error code: %s, error message: %s"
                        % (
                            response.request_id,
                            response.status_code,
                            response.code,
                            response.message,
                        ),
                    )
                    raise ValueError
        except Exception as e:
            logger.error(traceback.format_exc())
            return
    else:
        full_content = responses.output.choices[0].messages[0]["content"]
        request_id = responses.get("request_id", "")
        yield full_content + insert_request_id(request_id)


def send_feedback(data: gr.EventData) -> Any:
    """send feedback to server"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "secret-api-key-1")
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    body = data._data.get("value", {})
    print("feedback value", body)
    response = requests.post(
        url=feedback_service_url,
        headers=headers,
        data=body,
    )
    print("feedback response", response)
    return response


preloaded_messages = [
    (
        None,
        {
            "text": "这里是AgentScope的答疑助手，您有什么问题吗？",
            "files": None,
        },
    ),
]


class MyChatInterface(gr.ChatInterface):
    """self-define chat interface with feedback"""

    def _append_multimodal_history(
        self,
        message: Any,
        response: Any,
        history: Any,
    ) -> Any:
        if self.type == "tuples":
            images = []
            for x in message.files:
                path = x.path
                # path = path.replace(" ", "%20")
                images.append(path)
                print(path, x.path)
            # image_str = " ".join(f'\n ![image]({path})' for path in images)
            image_str = " ".join(f'\n <img src="{path}">' for path in images)
            if message.text is None or not isinstance(message.text, str):
                history.append([image_str, None])
                return
            elif message.text == "" and message.files != []:
                history.append([None, response])  # type: ignore
            else:
                message.text += image_str
                history.append([message.text, cast(str, response)])  # type: ignore
        else:
            for x in message.files:
                history.append(
                    {"role": "user", "content": cast(FileDataDict, x.model_dump())},  # type: ignore
                )
            if message.text is None or not isinstance(message.text, str):
                return
            else:
                history.append({"role": "user", "content": message.text})  # type: ignore
            if response:
                history.append(cast(MessageDict, response))


with gr.Blocks() as demo:
    mgr_chatbot = mgr.Chatbot(
        flushing=False,
        height=600,
        sanitize_html=False,
        custom_components={
            "answer-input": {
                "props": ["id"],
                "js": answer_input_js,
            },
        },
        value=preloaded_messages,
        avatar_images=[
            os.path.join(
                os.path.dirname(__file__),
                "./user.jpeg",
            ),
            os.path.join(
                os.path.dirname(__file__),
                "./bot.jpeg",
            ),
        ],
        # data_preprocess=fn,
        # data_postprocess=post_fn,
    )
    mgr_chatbot.type = "tuples"
    mgr_chatbot.custom(send_feedback)
    # user_input = mgr.MultimodalInput()
    # user_input.submit(fn=chat,
    #                   inputs=[user_input, mgr_chatbot],
    #                   outputs=[user_input, mgr_chatbot])
    demo.load()
    final_chatbot = MyChatInterface(
        fn=chat,
        submit_btn="发送",
        stop_btn="停止",
        retry_btn="🔄 重试",
        undo_btn="↩️ 撤回",
        clear_btn="🗑️ 清空",
        chatbot=mgr_chatbot,
        examples=[
            {"text": "介绍一下AgentScope？"},
            {"text": "AgentScope如何使用大模型？"},
            {"text": "AgentScope有什么样例？"},
        ],
        cache_examples=False,
        multimodal=True,
        textbox=gr.MultimodalTextbox(),
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()
