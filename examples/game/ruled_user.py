# -*- coding: utf-8 -*-
import time
import json
import random
from enum import Enum
from typing import Optional, Union, Any, Callable

import inquirer

from agentscope.agents import AgentBase
from agentscope.message import Msg
from utils import (
    send_chat_msg,
    query_answer,
    get_player_input,
    ResetException,
)


class FoodQuality(Enum):
    DELICIOUS = "美味的"
    BURNT = "烧焦的"
    FRESH = "新鲜的"
    STALE = "不新鲜的"
    SPICY = "辣的"
    BLAND = "清淡的"
    SWEET = "甜的"
    SOUR = "酸的"
    BITTER = "苦的"
    SALTY = "咸的"
    SAVORY = "鲜美的"
    CRISPY = "脆的"
    TENDER = "嫩的"
    JUICY = "多汁的"
    RICH = "浓郁的"
    FRAGRANT = "香的"
    SMOKY = "烟熏味的"


class RuledUser(AgentBase):
    """User agent under rules"""

    def __init__(
        self,
        name: str = "User",
        model: Optional[Union[Callable[..., Any], str]] = None,
        sys_prompt: Optional[str] = None,
        ingredients_dict: Optional[dict] = None,
        cook_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a RuledUser object."""
        super().__init__(name=name, model=model, sys_prompt=sys_prompt)
        self.retry_time = 3
        self.ingredients_dict = ingredients_dict
        self.cook_prompt = cook_prompt
        self.uid = kwargs.get("uid", None)

    def reply(
        self,
        x: dict = None,
        required_keys: Optional[Union[list[str], str]] = None,
    ) -> dict:
        """
        Processes the input provided by the user and stores it in memory,
        potentially formatting it with additional provided details.
        """
        if x is not None:
            self.memory.add(x)

        # TODO: To avoid order confusion, because `input` print much quicker
        #  than logger.chat
        time.sleep(0.5)
        while True:
            try:
                content = get_player_input(self.name, uid=self.uid)
                if x == {"content": "游戏开始"} and content == "":
                    send_chat_msg("【系统】有顾客光临，请接待。", uid=self.uid)
                    continue
                elif isinstance(x, dict):
                    if x.get("content") == "今天老板邀请大家一起来聚聚。" and content == "":
                        content = "大家好"
                    elif x.get("content") == "自定义输入" and content == "":
                        content = "你好"

                if not hasattr(self, "model") or len(content) == 0:
                    break

                ruler_res = self.is_content_valid(content)
                if ruler_res.get("allowed") == "true":
                    break

                send_chat_msg(
                    f"【系统】输入被规则禁止"
                    f" {ruler_res.get('reason', 'Unknown reason')}\n"
                    f"请重试",
                    "⚠️",
                    uid=self.uid,
                )
            except ResetException:
                raise ResetException
            except Exception as e:
                send_chat_msg(f"【系统】无效输入 {e}\n 请重试", "⚠️", uid=self.uid)

        kwargs = {}
        if required_keys is not None:
            if isinstance(required_keys, str):
                required_keys = [required_keys]

            for key in required_keys:
                kwargs[key] = get_player_input(key, uid=self.uid)

        if content == "做菜":
            content = self.cook()
            kwargs["food"] = content

        # Add additional keys
        msg = Msg(
            self.name,
            role="user",
            content=content,
            **kwargs,  # type: ignore[arg-type]
        )

        # Add to memory
        self.memory.add(msg)

        return msg

    def is_content_valid(self, content):
        prompt = self.sys_prompt.format_map({"content": content})
        message = Msg(name="user", content=prompt, role="user")
        ruler_res = self.model(
            messages=[message],
            parse_func=json.loads,
            max_retries=self.retry_time,
        )
        return ruler_res

    def set_ingredients(self, ingredients_dict):
        self.ingredients_dict = ingredients_dict

    def cook(self):
        ingredients_list = [
            item
            for sublist in self.ingredients_dict.values()
            for item in sublist
        ]

        cook_list = []
        questions = [
            inquirer.List(
                "ingredient",
                message="以下是今天拥有的食材是，请选择您需要使用的食材，完成后按回车键",
                choices=ingredients_list,
            ),
        ]

        choose_ingredient = f"""【系统】请选择需要的食材: <select-box shape="card"
                     type="checkbox" item-width="1000"
                    options='{json.dumps(ingredients_list)}' select-once
                    submit-text="确定"></select-box>"""

        send_chat_msg(
            choose_ingredient,
            flushing=False,
            uid=self.uid,
        )
        while True:
            sel_ingr = query_answer(questions, "ingredient", uid=self.uid)
            if isinstance(sel_ingr, str):
                send_chat_msg("【系统】请在列表中进行选择。", uid=self.uid)
                continue
            cook_list = sel_ingr
            break

        prompt = self.cook_prompt.format_map(
            {"ingredient": "+".join(cook_list)},
        )
        message = Msg(name="user", content=prompt, role="user")
        food = self.model(messages=[message])
        # random_quality = random.choice(list(FoodQuality)).value
        # food = random_quality + food

        send_chat_msg(
            # f"【系统】魔法锅周围光芒四射，你听到了轻微的咔哒声。当一切平静下来，一道《{food}》出现在你眼前。",
            f"【系统】你开始撸起袖子、开启炉灶。。。。当一切平静下来，一道《{food}》出现在客人眼前。",
            uid=self.uid,
        )

        return food
