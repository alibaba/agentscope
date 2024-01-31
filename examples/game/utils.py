# -*- coding: utf-8 -*-
import os
import pickle
from http import HTTPStatus
from typing import Optional, List
from datetime import datetime

import base64
import dashscope
import time

import requests
from colorist import BgBrightColor
import inquirer
import random
from multiprocessing import Queue, Value
from collections import defaultdict
from dataclasses import dataclass
from agentscope.message import Msg
from enums import StagePerNight
from pathlib import Path
from pypinyin import lazy_pinyin, Style

SYS_MSG_PREFIX = '【系统】'
DEFAULT_AGENT_IMG_DIR = "/tmp/as_game/img"
OPENING_ROUND = 3
REVISION_ROUND = 3

USE_WEB_UI = False


class GameCheckpoint:
    def __init__(
        self,
        stage_per_night: StagePerNight,
        customers: list,
        cur_plots: list[int],
        all_plots: dict,
        invited_customers: list[str],
        visit_customers: list,
    ):
        self.stage_per_night = stage_per_night
        self.customers = customers
        self.cur_plots = cur_plots
        self.all_plots = all_plots
        self.invited_customers = invited_customers
        self.visit_customers = visit_customers


def save_game_checkpoint(
    checkpoint: GameCheckpoint,
    checkpoint_prefix: str,
) -> None:
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_prefix + time_str
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_game_checkpoint(checkpoint_path: str) -> GameCheckpoint:
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def speak_print(m: Msg):
    print(f"{BgBrightColor.BLUE}{m.name}{BgBrightColor.OFF}: {m.content}")


def get_avatar_files(assets_path="assets"):
    files = Path(assets_path).glob("*avatar*")
    return [str(file) for file in files]


def get_a_random_avatar():
    return random.choices(get_avatar_files())


def get_use_web_ui():
    global USE_WEB_UI
    return USE_WEB_UI


def disable_web_ui():
    global USE_WEB_UI
    USE_WEB_UI = False


def enable_web_ui():
    global USE_WEB_UI
    USE_WEB_UI = True


def init_uid_queues():
    return {
        "glb_queue_chat_msg": Queue(),
        "glb_queue_chat_input": Queue(),
        "glb_queue_clue": Queue(),
        "glb_queue_story": Queue(),
    }


glb_uid_dict = defaultdict(init_uid_queues)


def send_chat_msg(
    msg,
    role=None,
    uid=None,
    flushing=False,
    avatar="./assets/bot.jpg",
):
    print("send_chat_msg:", msg)
    if get_use_web_ui():
        global glb_uid_dict
        glb_queue_chat_msg = glb_uid_dict[uid]["glb_queue_chat_msg"]
        glb_queue_chat_msg.put(
            [
                None,
                {
                    "text": msg,
                    "name": role,
                    "flushing": flushing,
                    "avatar": avatar,
                },
            ],
        )


def send_clue_msg(
    clue,
    unexposed_num=0,
    role=None,
    uid=None,
):
    print("send_clue_msg:", clue)
    if get_use_web_ui():
        global glb_uid_dict
        glb_queue_clue = glb_uid_dict[uid]["glb_queue_clue"]
        glb_queue_clue.put(
            {
                "clue": clue,
                "name": role,
                "unexposed_num": unexposed_num,
            }
        )


def get_clue_msg(
    uid=None,
):
    global glb_uid_dict
    glb_queue_clue = glb_uid_dict[uid]["glb_queue_clue"]
    if not glb_queue_clue.empty():
        line = glb_queue_clue.get(block=False)
        if line is not None:
            return line
    return None


def send_story_msg(
    story,
    role=None,
    uid=None,
):
    print("send_story_msg:", story)
    if get_use_web_ui():
        global glb_uid_dict
        glb_queue_story = glb_uid_dict[uid]["glb_queue_story"]
        glb_queue_story.put(
            {
                "story": story,
                "name": role,
            }
        )


def get_story_msg(
    uid=None,
):
    global glb_uid_dict
    glb_queue_story = glb_uid_dict[uid]["glb_queue_story"]
    if not glb_queue_story.empty():
        line = glb_queue_story.get(block=False)
        if line is not None:
            return line
    return None


def send_player_msg(
    msg,
    role="我",
    uid=None,
    flushing=False,
    avatar="./assets/user.jpg",
):
    print("send_player_msg:", msg)
    if get_use_web_ui():
        global glb_uid_dict
        glb_queue_chat_msg = glb_uid_dict[uid]["glb_queue_chat_msg"]
        glb_queue_chat_msg.put(
            [
                {
                    "text": msg,
                    "name": role,
                    "flushing": flushing,
                    "avatar": avatar,
                },
                None,
            ],
        )


def get_chat_msg(uid=None):
    global glb_uid_dict
    glb_queue_chat_msg = glb_uid_dict[uid]["glb_queue_chat_msg"]
    if not glb_queue_chat_msg.empty():
        line = glb_queue_chat_msg.get(block=False)
        if line is not None:
            return line
    return None


def send_player_input(msg, role="餐厅老板", uid=None):
    if get_use_web_ui():
        global glb_uid_dict
        glb_queue_chat_input = glb_uid_dict[uid]["glb_queue_chat_input"]
        glb_queue_chat_input.put([None, msg])


def send_pretty_msg(msg, uid=None, flushing=True, avatar="./assets/bot.jpg"):
    speak_print(msg)
    if get_use_web_ui():
        global glb_uid_dict
        send_chat_msg(
            msg.content,
            uid=uid,
            role=msg.name,
            flushing=flushing,
            avatar=avatar,
        )


def get_player_input(name=None, uid=None):
    global glb_uid_dict
    if get_use_web_ui():
        print("wait queue input")
        glb_queue_chat_input = glb_uid_dict[uid]["glb_queue_chat_input"]
        content = glb_queue_chat_input.get(block=True)[1]
        print(content)
        if content == "**Reset**":
            glb_uid_dict[uid] = init_uid_queues()
            raise ResetException
    else:
        content = input(f"{name}: ")
    return content


def format_choices(choices):
    formatted_choices = ""
    line_length = 0

    for index, choice in enumerate(choices):
        choice_str = f"[{index + 1}]. {choice}  "
        choice_length = len(choice_str)

        if line_length + choice_length > 30:
            formatted_choices += "\n"
            line_length = 0

        formatted_choices += choice_str
        line_length += choice_length

    formatted_choices = formatted_choices.rstrip()

    return formatted_choices


def query_answer(questions: List, key="ans", uid=None):
    if get_use_web_ui():
        return get_player_input(uid=uid)
    else:
        answer = [inquirer.prompt(questions)[key]]  # return list
    return answer


@dataclass
class CheckpointArgs:
    load_checkpoint: str = None
    save_checkpoint: str = "./checkpoints/cp-"


class ResetException(Exception):
    pass


def generate_picture(prompt, model="wanx-lite"):
    from dashscope.common.error import InvalidTask
    dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY") or dashscope.api_key
    assert dashscope.api_key
    try:
        if model == "wanx-lite":
            rsp = dashscope.ImageSynthesis.call(
                model='wanx-lite',
                prompt=prompt,
                n=1,
                size='768*768')
        else:
            rsp = dashscope.ImageSynthesis.call(
                model=dashscope.ImageSynthesis.Models.wanx_v1,
                prompt=prompt,
                n=1,
                size='1024*1024')
        if rsp.status_code == HTTPStatus.OK:
            return rsp.output['results'][0]['url']

        else:
            print('Failed, status_code: %s, code: %s, message: %s' %
                  (rsp.status_code, rsp.code, rsp.message))
    except InvalidTask as e:
        print(e)


def get_clue_image_b64_url(customer, clue_name, uid, content):
    prompt = """
    Create an abstract, comic-style clue card for the theme {clue_name}. 
    The card should visually represent the concept of {content} in a way 
    that is stylistically exaggerated and colorful, typical of comic art. 
    The imagery should be engaging and easily decipherable, providing clear 
    hints related to the theme without being too literal. Use bold lines, 
    dynamic shapes, and vibrant colors to encapsulate the essence of the 
    clue in an artistic and abstract representation.
    """
    extensions = ["gif", "jpeg", "png", "jpg"]
    try:
        file_dir = os.path.join(DEFAULT_AGENT_IMG_DIR, uid, customer)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        file_path = None
        for ext in extensions:
            tmp_file_path = os.path.join(file_dir, f"{clue_name}.{ext}")
            if os.path.exists(tmp_file_path):
                file_path = tmp_file_path
                break
        if file_path is None:
            url = generate_picture(prompt.format_map({
                "clue_name": clue_name,
                "content": content,
            }))
            response = requests.get(url)
            if response.status_code == 200:
                for ext in extensions:
                    if f".{ext}" in url:
                        file_path = os.path.join(file_dir, f"{clue_name}.{ext}")
                        break
                if file_path:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise Exception(f"Unknown file extension: {url}")
            else:
                raise Exception(
                    f"Error downloading image: status code {response.status_code}")

        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            base64_data = encoded_string.decode("utf-8")
            base64_url = f"data:image/{ext};base64,{base64_data}"
        return base64_url
    except Exception as e:
        print(e)
        return "#"


def replace_names_in_messages(messages):
    for line in messages:
        if 'name' in line:
            name = line['name']
            if any('一' <= char <= '鿿' for char in name):
                # 将中文名字转换为带音调的拼音
                pinyin_name = ''.join(lazy_pinyin(name, style=Style.TONE3))
                line['name'] = pinyin_name
    return messages
