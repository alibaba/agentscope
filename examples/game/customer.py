# -*- coding: utf-8 -*-
import copy
from typing import Any, Union, Tuple
import re
import json
import numpy as np
from loguru import logger

from enums import CustomerConv, CustomerPlot
from agentscope.agents import StateAgent, DialogAgent
from agentscope.message import Msg
from relationship import Relationship
from utils import (
    send_chat_msg,
    send_clue_msg,
    get_a_random_avatar,
    send_pretty_msg,
    replace_names_in_messages,
    SYS_MSG_PREFIX,
    get_clue_image_b64_url,
    extract_keys_from_dict,
    send_story_msg,
)

HISTORY_WINDOW = 10
# TODO: for debug, set the score bars to be lower
MIN_BAR_RECEIVED_CONST = 4
MESSAGE_KEYS = ["name", "role", "content"]


class Customer(StateAgent, DialogAgent):
    def __init__(self, game_config: dict, **kwargs: Any):
        self.uid = kwargs.pop("uid")
        super().__init__(**kwargs)
        self.retry_time = 3
        self.game_config = game_config
        self.max_itr_preorder = 5
        self.preorder_itr_count = 0
        self.avatar = self.config.get("avatar", get_a_random_avatar())
        self.background = self.config["character_setting"]["background"]
        # self.friendship = int(self.config.get("friendship", 60))
        self.is_satisfied = False
        self.relationship = Relationship(
            self.config.get("relationship", "陌生"),
            MIN_BAR_RECEIVED_CONST,
        )
        self.preferred_info = ''
        self.cur_state = CustomerConv.WARMING_UP
        # TODO: A customer can be in at most one plot in the current version
        self.active_plots = []
        self.prev_active_plots = []

        self.register_state(
            state=CustomerConv.OPENING,
            handler=self._opening_chat,
        )

        self.register_state(
            state=CustomerConv.WARMING_UP,
            handler=self._pre_meal_chat,
        )
        self.register_state(
            state=CustomerConv.AFTER_MEAL_CHAT,
            handler=self._main_plot_chat,
        )
        self.register_state(
            state=CustomerConv.INVITED_GROUP_PLOT,
            handler=self._main_plot_chat,
        )

        # TODO: refactor to a sub-state
        self.plot_stage = CustomerPlot.NOT_ACTIVE

        # Clues: `unexposed_clues` & `exposed_clues`
        self.unexposed_clues = self.config.get("clue", None)
        # if self.unexposed_clues is None:
        #     self.unexposed_clues = self.build_clues()
        #     self.config['clue'] = copy.deepcopy(self.unexposed_clues)

        # print(self.unexposed_clues)

        self.hidden_plot = {}
        for item in self.unexposed_clues:
            if item["plot"] in self.hidden_plot.keys():
                self.hidden_plot[item["plot"]] += item["content"]
            else:
                self.hidden_plot[item["plot"]] = item["content"]

        # For initialization
        send_clue_msg(
            None,
            unexposed_num=len(self.unexposed_clues),
            uid=self.uid,
            role=self.name,
        )
        self.exposed_clues = []

    def visit(self) -> np.array:
        # TODO: for debug, set the visit prob to be 0.9
        return np.random.binomial(n=1, p=0.9) > 0
        # return (
        #     np.random.binomial(
        #         n=1,
        #         p=min(self.friendship / 100, 1.0),
        #     )
        #     > 0
        # )

    def activate_plot(self, active_plots: list[int]) -> None:
        # when the customer is the main role in a plot, it will be activated
        self.plot_stage = CustomerPlot.ACTIVE
        for p in active_plots:
            logger.debug(f"plot {p}, {active_plots}")
            if (
                p in self.hidden_plot
                and len(self.active_plots) == 0
            ):
                self.active_plots = [p]
            elif p in self.hidden_plot:
                raise ValueError(
                    "A customer can be in at most one plot in the current "
                    "version",
                )
            else:
                logger.error(f"Plot {p} is not defined in the hidden_plot")
                raise ValueError

    def deactivate_plot(self) -> None:
        # when the plot in which the customer is a main role is over, the
        # customer will be deactivated
        self.plot_stage = CustomerPlot.NOT_ACTIVE
        self.prev_active_plots = [self.active_plots[0]]
        self.active_plots = []


    def reply(self, x: dict = None) -> Union[dict, tuple]:
        # TODO:
        # not sure if it is some implicit requirement of the tongyi chat api,
        # the first/last message must have role 'user'.
        if x is not None:
            x["role"] = "user"
        logger.debug(
            f"{self.name} state: {self.cur_state} {self.plot_stage}"
            f" {self.active_plots}",
        )
        send_chat_msg("**speak**", role=self.name, uid=self.uid,
                      avatar=self.avatar)
        msg = StateAgent.reply(self, x=x)
        send_pretty_msg(msg, uid=self.uid, avatar=self.avatar)
        return msg

    def _recommendation_to_score(self, x: dict) -> dict:
        food = x["content"]
        food_judge_prompt = self.game_config["food_judge_prompt"]
        food_judge_prompt = food_judge_prompt.format_map(
            {
                "food_preference": self.config["character_setting"][
                    "food_preference"
                ],
                'preferred_info': self.preferred_info,
                "food": food,
            },
        )
        message = Msg(name="user", content=food_judge_prompt, role="user")

        def _parse_score(text: Any) -> Tuple[float, Any]:
            score = re.search("([0-9]+)分", str(text)).groups()[0]
            return float(score), text

        def _default_score(_: str) -> float:
            return 2.0

        score, text = self.model(
            [extract_keys_from_dict(message, MESSAGE_KEYS)],
            parse_func=_parse_score,
            fault_handler=_default_score,
            max_retries=3,
        )

        satisfied_str = "不满意"
        is_satisfied = False
        if self.relationship.is_satisfied(score):
            satisfied_str = "满意"
            is_satisfied = True
        text = satisfied_str + "，" + text

        prev_relationship = self.relationship.to_string()
        self.relationship.update(score)
        cur_relationship = self.relationship.to_string()
        chat_text = f" {SYS_MSG_PREFIX} {self.name}感觉{food}{satisfied_str}, "

        if prev_relationship != cur_relationship:
            chat_text += f"你们的关系从{prev_relationship}变得{cur_relationship}了"
        else:
            chat_text += f"你们的关系没变化，依旧是{prev_relationship}。"

        send_chat_msg(
            chat_text,
            uid=self.uid,
        )

        if is_satisfied or (
            score >= MIN_BAR_RECEIVED_CONST
            # and self.friendship >= MIN_BAR_FRIENDSHIP_CONST
        ):
            self.transition(CustomerConv.AFTER_MEAL_CHAT)
            print("---", self.cur_state)
        self.preorder_itr_count = 0

        return Msg(
            role="assistant",
            name=self.name,
            content=text,
            score=score,
            is_satisfied=is_satisfied,
            relationship=cur_relationship,
        )

    def _opening_chat(self, x: dict) -> dict:
        system_prompt = self.game_config["basic_background_prompt"].format_map(
            {
                "name": self.config["name"],
                "character_description": self.background
            }
        ) + self.game_config[
                            "hidden_main_plot_prompt"
                        ].format_map(
            {
                "hidden_plot": self.hidden_plot[self.active_plots[0]],
            },
        )
        if x is not None:
            self.memory.add(x)

        system_msg = Msg(role="user", name="system", content=system_prompt)
        prompt = self.engine.join(
            self._validated_history_messages(recent_n=HISTORY_WINDOW),
            system_msg,
            x,
        )

        logger.debug(prompt)

        reply = self.model(replace_names_in_messages(prompt))
        reply_msg = Msg(role="assistant", name=self.name, content=reply)
        self.memory.add(reply_msg)

        self.update_clues(reply_msg.content)

        return reply_msg

    def _preferred_food(self, x:dict) -> dict:
        ingredients_dict = x['content']
        # breakpoint()
        ingredients_list = [
            item
            for sublist in ingredients_dict.values()
            for item in sublist
        ]
        ingredients = "、".join(ingredients_list)

        system_prompt = self.game_config["preferred_food_prompt"].format_map(
            {
                "name": self.config["name"],
                "food_preference": self.config["character_setting"]["food_preference"],
                "ingredients": ingredients,
            },
        )
        system_msg = Msg(role="user", name="system", content=system_prompt)
        # prepare prompt
        prompt = self.engine.join(
            self._validated_history_messages(recent_n=HISTORY_WINDOW),
            system_msg)
        logger.debug(system_prompt)

        reply = self.model(replace_names_in_messages(prompt))
        self.preferred_info = reply
        reply_msg = Msg(role="assistant", name=self.name, content=reply)
        self.memory.add(reply_msg)
        return reply_msg

    def _pre_meal_chat(self, x: dict) -> dict:
        if "food" in x:
            return self._recommendation_to_score(x)

        return self._preferred_food(x)

    def _main_plot_chat(self, x: dict) -> dict:
        """
        _main_plot_chat
        :param x:
        :return:
        Stages of the customer defines the prompt past to the LLM
        1. Customer is a main role in the current plot
            1.1 the customer has hidden plot
            1.2 the customer has no hidden plot (help with background)
        2. Customer is not a main role in the current plot
        """
        prompt = self._gen_plot_related_prompt()

        logger.debug(f"{self.name} system prompt: {prompt}")

        system_msg = Msg(role="user", name="system", content=prompt)

        join_args = [
            self._validated_history_messages(recent_n=HISTORY_WINDOW),
            system_msg,
        ]

        if x is not None:
            join_args.append(x)
            self.memory.add(x)

        prompt = self.engine.join(*join_args)

        logger.debug(f"{self.name} history prompt: {prompt}")

        reply = self.model(replace_names_in_messages(prompt))

        reply_msg = Msg(role="assistant", name=self.name, content=reply)
        self.memory.add(reply_msg)

        self.update_clues(reply_msg.content)

        return reply_msg

    def refine_background(self) -> None:
        background_prompt = self.game_config[
            "basic_background_prompt"
        ].format_map(
            {
                "name": self.config["name"],
                "character_description": self.background,
            },
        )
        background_prompt += self.game_config[
            "hidden_main_plot_prompt"
        ].format_map(
            {
                "hidden_plot": self.config["character_setting"][
                    "hidden_plot"
                ][self.prev_active_plots[0]],
            },
        )
        analysis_prompt = background_prompt + self.game_config["analysis_conv"]

        system_msg = Msg(role="user", name="system", content=analysis_prompt)

        prompt = self.engine.join(
            self._validated_history_messages(recent_n=HISTORY_WINDOW * 2),
            system_msg,
        )

        analysis = self.model(replace_names_in_messages(prompt))
        analysis_msg = Msg(
            role="user",
            name=self.name,
            content=f"聊完之后，{self.name}在想:" + analysis,
        )
        send_pretty_msg(
            analysis_msg,
            uid=self.uid,
            avatar=self.avatar,
        )

        update_prompt = self.game_config["update_background"].format_map(
            {
                "analysis": analysis,
                "background": self.background,
                "name": self.name,
            },
        )
        update_msg = Msg(role="user", name="system", content=update_prompt)
        new_background = self.model(
            [extract_keys_from_dict(update_msg, MESSAGE_KEYS)]
        )

        bg_msg = f" {SYS_MSG_PREFIX}根据对话，{self.name}的背景更新为：" + new_background
        send_chat_msg(bg_msg, uid=self.uid, flushing=True)

        self.background = new_background

    def _validated_history_messages(self, recent_n: int = 10):
        hist_mem = self.memory.get_memory(recent_n=recent_n)
        if len(hist_mem) > 0:
            hist_mem[0]["role"], hist_mem[-1]["role"] = "user", "user"
        return hist_mem

    def generate_pov_story(self, recent_n: int = 20) -> None:
        related_mem = self._validated_history_messages(recent_n)
        conversation = ""
        for mem in related_mem:
            if "name" in mem:
                conversation += mem["name"] + ": " + mem["content"]
            else:
                conversation += "背景" + ": " + mem["content"]
        background = self.background
        if self.plot_stage == CustomerPlot.ACTIVE:
            background += self.hidden_plot[self.active_plots[0]]

        pov_prompt = self.game_config["pov_story"].format_map(
            {
                "name": self.name,
                "background": background,
                "conversation": conversation,
            },
        )
        msg = Msg(name="system", role="user", content=pov_prompt)
        send_chat_msg(
            "**speak**",
            role=self.name,
            uid=self.uid,
            avatar=self.avatar,)
        pov_story = self.model(
            [extract_keys_from_dict(msg, MESSAGE_KEYS)]
        )
        send_story_msg(pov_story, role=self.name, uid=self.uid)
        print("*" * 20)
        send_chat_msg(
            pov_story,
            role=self.name,
            uid=self.uid,
            avatar=self.avatar,
        )
        print("*" * 20)

    def _gen_plot_related_prompt(self) -> str:
        """
        generate prompt depending on the state and friendship of the customer
        """
        prompt = self.game_config["basic_background_prompt"].format_map(
            {
                "name": self.config["name"],
                "character_description": self.background,
            },
        )

        if (
            self.plot_stage == CustomerPlot.ACTIVE
        ):
            # get the clues related to the current plot
            curr_clues = []
            for c in self.config["clue"]:
                if c["plot"] == self.active_plots[0]:
                    curr_clues.append(c)
            # compose the clues according the relationship level
            if not self.relationship.is_max():
                end_idx = len(curr_clues) // 3 * \
                          self.relationship.level.value
                hidden_plot = "\n".join(
                    [c["content"] for c in curr_clues[:end_idx]])
            else:
                hidden_plot = "\n".join(
                    [c["content"] for c in curr_clues])
            # -> prompt for the main role in the current plot
            prompt += self.game_config["hidden_main_plot_prompt"].format_map(
                {
                    "hidden_plot": hidden_plot,
                },
            )
            if self.cur_state == CustomerConv.AFTER_MEAL_CHAT:
                prompt += self.game_config["hidden_main_plot_after_meal"]
            else:
                prompt += self.game_config["hidden_main_plot_discussion"]
        else:
            # -> prompt for the helper or irrelvant roles in the current plot
            if self.cur_state == CustomerConv.AFTER_MEAL_CHAT:
                prompt += self.game_config["regular_after_meal_prompt"]
            else:
                prompt += self.game_config["invited_chat_prompt"]

        prompt += self.game_config[self.relationship.prompt]
        logger.debug(prompt)
        return prompt

    def talk(self, content, is_display=True, flushing=True):
        if content is not None:
            msg = Msg(
                role="user",
                name=self.name,
                content=content,
            )
            self.memory.add(msg)
            if is_display:
                send_chat_msg(
                    content,
                    role=self.name,
                    uid=self.uid,
                    avatar=self.avatar,
                    flushing=flushing,
                )
            return msg

    def build_clues(self):
        # Get all hidden plot
        send_chat_msg(f"{SYS_MSG_PREFIX}初始化NPC {self.name}..."
                      f"（这可能需要一些时间）", uid=self.uid)

        clues = []
        for i, plot in self.hidden_plot.items():
            clue_parse_prompt = self.game_config["clue_parse_prompt"] + plot
            message = Msg(name="system", role="user", content=clue_parse_prompt)

            curr_clues = self.model(
                [extract_keys_from_dict(message, MESSAGE_KEYS)],
                parse_func=json.loads,
                max_retries=self.retry_time,
            )
            for c in curr_clues:
                c["plot"] = i
                clues.append(c)
        logger.debug(clues)
        send_chat_msg(f"{SYS_MSG_PREFIX}初始化NPC {self.name}完成！", uid=self.uid)
        return clues

    def update_clues(self, content):

        if len(self.unexposed_clues) == 0:
            return

        prompt = self.game_config["clue_detect_prompt"].format_map(
            {
                "content": content,
                "clue": self.unexposed_clues,
                "name": self.name,
            }
        )
        message = Msg(name="system", content=prompt, role="user")
        exposed_clues = self.model(
            [extract_keys_from_dict(message, MESSAGE_KEYS)],
            parse_func=json.loads,
            fault_handler=lambda response: [],
            max_retries=self.retry_time,
        )
        logger.debug(exposed_clues)
        logger.debug(self.unexposed_clues)
        indices_to_pop = []
        found_clue = []
        if not isinstance(exposed_clues, list):
            return

        for clue in exposed_clues:
            if not isinstance(clue, dict):
                continue
            index = clue.get("index", -1)
            summary = clue.get("summary", -1)
            if len(self.unexposed_clues) > index >= 0:
                indices_to_pop.append(index)
                # TODO: get index and summary separately can be more stable
                found_clue.append(
                    {
                        "name": self.unexposed_clues[index]["name"],
                        "summary": summary,  # Use new summary
                        "image": get_clue_image_b64_url(
                            customer=self.name,
                            clue_name=self.unexposed_clues[index]["name"],
                            uid=self.uid,
                            content=summary,
                        )
                    }
                )
        indices_to_pop.sort(reverse=True)
        logger.debug(indices_to_pop)
        for index in indices_to_pop:
            element = self.unexposed_clues.pop(index)
            self.exposed_clues.append(element)

        for i, clue in enumerate(found_clue):
            send_chat_msg(
                f"{SYS_MSG_PREFIX}发现{self.name}的新线索（请查看线索栏）："
                f"《{clue['name']}》{clue['summary']} "
                f"\n\n剩余未发现线索数量:"
                f"{len(self.unexposed_clues) + len(found_clue) - i - 1}",
                uid=self.uid)
            send_clue_msg(
                clue,
                unexposed_num=len(self.unexposed_clues),
                uid=self.uid,
                role=self.name,
            )