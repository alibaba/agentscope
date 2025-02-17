# -*- coding: utf-8 -*-
# pylint: disable=C0301
"""
This script defines all the agents used in the data interpreter pipeline.
"""
from typing import Any, Dict, List, Tuple, Optional, Union, Sequence
from agentscope.agents import ReActAgent, DialogAgent
from agentscope.message import Msg
from agentscope.models import ModelResponse
from agentscope.parsers import MarkdownJsonObjectParser
from agentscope.parsers import RegexTaggedContentParser
from agentscope.service import ServiceToolkit


class PlannerAgent(DialogAgent):
    """
    PlannerAgent is responsible for decomposing complex tasks into manageable
    subtasks.

    This agent takes an overall task and breaks it down into subtasks that can
    be solved using available tools or code execution. It ensures that each
    subtask is appropriately sized and prioritizes using tools over code
    execution when possible.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        service_toolkit: ServiceToolkit,
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        self.service_toolkit = service_toolkit

    def __call__(self, x: Msg) -> List[Dict[str, Any]]:
        return self.plan(x)

    def plan(self, x: Msg) -> List[Dict[str, Any]]:
        """
        Decompose the task provided in the message into subtasks.

        Args:
            x (Msg): Message containing the task to be decomposed.

        Returns:
            List[Dict[str, Any]]: List of subtasks as dictionaries.
        """
        messages = x.content
        subtasks = self._decompose_task(messages)
        return subtasks

    def _decompose_task(
        self,
        task: str,
        max_tasks: int = 5,
    ) -> List[Dict[str, Any]]:
        # Implement task decomposition
        message = Msg(
            name="User",
            role="user",
            content=f"""
                    Task: {task}
                    - Given the task above, break it down into subtasks with dependencies if it is sufficently complex to be solved in one go.
                    - Every subtask should be solvable through either executing code or using tools. The information of all the tools available are here:
                    {self.service_toolkit.tools_instruction}
                    - The subtask should not be too simple. If a task can be solve with a single block of code in one go, it should not be broken down further. Example: a subtask cannot be simply installing or importing libraries.
                    - Prioritze using other tools over `execute_python_code` and take the tools available into consideration when decomposing the task. Provide a JSON structure with the following format for the decomposition:
                        ```json
                        [
                            {{
                                "task_id": str = "unique identifier for a task in plan, can be an ordinal",
                                "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
                                "instruction": "what you should do in this task, one short phrase or sentence",
                                "task_type": "type of this task",
                                "tool_info": "recommended tool(s)' name(s) for solving this task",
                            }},
                            ...
                        ]
                        ```
                    - The maximum number of subtasks allowed is {max_tasks}.
                    """,
        )
        response_text: str = self.reply(message).content.strip()
        response = ModelResponse(text=response_text)
        parser = MarkdownJsonObjectParser()
        parsed_response: List[Dict[str, Any]] = parser.parse(response)
        return parsed_response.parsed


class VerifierAgent(ReActAgent):
    """
    VerifierAgent verifies if a given result successfully solves a subtask.

    This agent checks the result of a subtask execution to ensure it meets the
    requirements of the current subtask. It uses reasoning and available tools
    to perform the verification and reports whether the subtask is solved.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        service_toolkit: ServiceToolkit,
        *,
        max_iters: int = 10,
        verbose: bool = True,
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            service_toolkit=service_toolkit,
            max_iters=max_iters,
            verbose=verbose,
        )

        # Overwrite the parser attribute with the custom format_instruction to reinforce the output adhere to json format.
        self.parser = RegexTaggedContentParser(
            format_instruction="""Respond with specific tags as outlined below in json format:
        <thought>{what you thought}</thought>
        <function>{the function name you want to call}</function>
        <{argument name}>{argument value}</{argument name}>
        <{argument name}>{argument value}</{argument name}>
        ...""",  # noqa
            try_parse_json=True,
            required_keys=["thought", "function"],
        )

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        Verification_PROMPT = """- Given `overall_task` and `solved_dependent_sub_tasks` as context, verify if the information in `result` can succesfully solve `current_sub_task` with your reasoning trace.
        - If you think code or tools are helpful for verification, use `execute_python_code` and/or other tools available to do verification.
        - Do not simply trust the claim in `result`. VERIFY IT.
        - If the information in `result` cannot solve `current_sub_task`, Do NOT attempt to fix it. Report it IMMEDIATELY. You job is just to do the verification.
        - If the given result can succesfully solve `current_sub_task`, ALWAYS output 'True' at the very end of your response; otherwise, output why the given result cannot succesfully solve `current_sub_task` and followed by 'False'.
        - DO NOT call `finish` before the entire verification process is completed. After the entire verification is completed, use `finish` tool IMMEDIATELY."""

        msg = Msg(
            name="Verifier",
            role="system",
            content=Verification_PROMPT + x.content,
        )
        verdict = super().reply(msg)

        return verdict


class SynthesizerAgent(DialogAgent):
    """
    SynthesizerAgent combines the results of all subtasks to produce the final
    answer.

    This agent takes the overall task and the results of each solved subtask,
    synthesizes them, and generates a comprehensive answer for the overall task.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        Synthesize_PROMPT = """Given `overall_task` and all solved `subtasks`, synthesize the result of each tasks and give a answer for `overall_task`."""

        message = Msg(
            name="User",
            role="user",
            content=Synthesize_PROMPT + "  " + x.content,
        )
        final_msg = super().reply(message)

        return final_msg


class ReplanningAgent(DialogAgent):
    """
    ReplanningAgent handles replanning when a subtask cannot be solved as is.

    This agent decides whether to substitute an unsolvable subtask with a new one
    or to further decompose it into smaller subtasks. It then provides the
    updated plan or decomposition to continue solving the overall task.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        service_toolkit: ServiceToolkit,
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        self.service_toolkit = service_toolkit

    def __call__(self, x: Msg) -> Tuple[str, Any]:
        return self.replan(x)

    def replan(self, x: Msg) -> Tuple[str, Any]:
        """
        Decide to replan or decompose a subtask based on the given message.

        Args:
            x (Msg): Message containing the current task and context.

        Returns:
            Tuple[str, Any]: ('replan_subtask', new_tasks) or
                ('decompose_subtask', subtasks), depending on the decision.

        Raises:
            ValueError: If unable to determine how to revise the subtask.
        """

        task = x.content
        revising_PROMPT = """Based on `overall_task` and all solved `subtasks`, and the `VERDICT`, decide if it is better to :
        1. come out with another subtask in place of `current_sub_task` if you think the reason `current_sub_task` is unsolvable is it is infeasible to solve;
        2. further break `current_sub_task` into more subtasks if you think the reason `current_sub_task` is unsolvable is it is still too complex.
        If it is better to do '1', output 'replan_subtask'. If it is better to do '2', output 'decompose_subtask'."""
        message = Msg(
            name="User",
            role="user",
            content=revising_PROMPT + "  " + task,
        )
        option = self.reply(message).content.strip()
        print("replanning option: ", option)
        if "replan_subtask" in option:
            new_tasks = self._replanning(task)
            return ("replan_subtask", new_tasks)
        elif "decompose_subtask" in option:
            subtasks = self._decompose_task(task)
            return ("decompose_subtask", subtasks)
        else:
            raise ValueError("Not clear how to revise subtask.")

    def _replanning(self, task: str) -> List[Dict[str, Any]]:
        replanning_PROMPT = f"""Based on `overall_task` and all solved `subtasks`, and the `VERDICT`:
        1. Substitute `current_sub_task` with a new `current_sub_task` in order to better achieve `overall_task`.
        2. Modify all substasks that have dependency on `current_sub_task` based on the new `current_sub_task` if needed.
        3. Follow the format below to list your revised subtasks, including the solved subtasks:
        ```json
        [
            {{
                "task_id": str = "unique identifier for a task in plan, can be an ordinal",
                "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
                "instruction": "what you should do in this task, one short phrase or sentence",
                "task_type": "type of this task",
                "tool_info": "recommended tool(s)' name(s) for solving this task",
            }},
            ...
        ]
        ```
        4. Every new task/subtask should be solvable through either executing code or using tools. The information of all the tools available are here:
                    {self.service_toolkit.tools_instruction} """
        message = Msg(
            name="User",
            role="user",
            content=replanning_PROMPT + "  " + task,
        )
        response_text: str = self.reply(message).content.strip()
        response = ModelResponse(text=response_text)
        parser = MarkdownJsonObjectParser()
        parsed_response: List[Dict[str, Any]] = parser.parse(response)
        return parsed_response.parsed

    def _decompose_task(
        self,
        task: str,
        max_tasks: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Decompose a complex subtask into smaller, more manageable subtasks.

        Args:
            task (str): The task to be decomposed
            max_tasks (int, optional): Maximum number of subtasks allowed. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of decomposed subtasks as dictionaries
        """
        message = Msg(
            name="User",
            role="user",
            content=f"""
                    Task: {task}
                    - Given the task above which was determined to be too complex, break it down into smaller, more manageable subtasks.
                    - Every subtask should be solvable through either executing code or using tools. The information of all the tools available are here:
                    {self.service_toolkit.tools_instruction}
                    - The subtask should not be too simple. If a task can be solved with a single block of code in one go, it should not be broken down further.
                    - Prioritize using other tools over `execute_python_code` and take the tools available into consideration when decomposing the task.
                    - Provide a JSON structure with the following format for the decomposition:
                        ```json
                        [
                            {{
                                "task_id": str = "unique identifier for a task in plan, can be an ordinal",
                                "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
                                "instruction": "what you should do in this task, one short phrase or sentence",
                                "task_type": "type of this task",
                                "tool_info": "recommended tool(s)' name(s) for solving this task",
                            }},
                            ...
                        ]
                        ```
                    - The maximum number of subtasks allowed is {max_tasks}.
                    """,
        )

        response_text: str = self.reply(message).content.strip()
        response = ModelResponse(text=response_text)
        parser = MarkdownJsonObjectParser()
        parsed_response: List[Dict[str, Any]] = parser.parse(response)
        return parsed_response.parsed
