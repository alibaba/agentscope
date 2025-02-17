# -*- coding: utf-8 -*-
# pylint: disable=C0301
"""
This script demonstrates task solving with a data interpreter pipeline
formed by multiple agents, each specialized in one aspect of problem solving.
"""
import os
import copy
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv, find_dotenv
from di_agents import (
    PlannerAgent,
    VerifierAgent,
    SynthesizerAgent,
    ReplanningAgent,
)
import agentscope
from agentscope.agents import ReActAgent
from agentscope.message import Msg
from agentscope.service import (
    ServiceToolkit,
    execute_python_code,
    list_directory_content,
    get_current_directory,
    execute_shell_command,
)

from agentscope.parsers import RegexTaggedContentParser

# Global variables for agents with type annotations
planner_agent: PlannerAgent
solver_agent: ReActAgent
verifier_agent: VerifierAgent
synthesizer_agent: SynthesizerAgent
replanner_agent: ReplanningAgent

# Global variables for failure tracking
total_failure_count = 0
max_total_failures = 3  # Adjust as needed

_ = load_dotenv(find_dotenv())  # read local .env file

openai_api_key = os.getenv("OPENAI_API_KEY")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")


STRUCTURAL_PROMPT = """
        # overall_task: {overall_task}

        # solved_sub_tasks: {solved_sub_tasks}

        # current_sub_task: {current_sub_task}

        # Instruction
        - Conditioning on `overall_task` and `solved_sub_tasks`, solve `current_sub_task` with the appropriate tools provided. Note that you should only use `overall_task` and `solved_sub_tasks` as context as opposed to solving them. DO NOT attempt to solve `overall_task` or `solved_sub_tasks`.
        - When using tools, ALWAYS prioritize using the tool mentioned in `tool_info` over other tool or code for solving `current_sub_task`.
        - While some concise thoughts are helpful, code is required, unless other tools are used. If certain python libraries are not installed, use `execute_shell_command` to install them.
        - At each step, if some data is fetched/generated, it is a good practice to save it.

        # Output Instruction
        - Always output one and only one code block in your response. The code block must be self-contained, i.e., does not rely on previously generated code to be executed successfully, because the execution environments do not persist between calls of `execute_python_code`. Always use print statement on the final solution. E.g., if `res` is the final output, use `print(res)` at the end of your code. Output the code itself if the task at hand is to generate that code. After that, use `execute_python_code` to execute your code. Based on the result from code execution or tool using, determine if `current_sub_task` is solved.
        - After `current_sub_task` is solved, return explicitly the result(s) for `current_sub_task` that is/are needed in the subsequent subtasks. If certain code are needed for the subsequent tasks, OUTPUT THE COMPLETE CODE. If the code is long, save it in txt or json format, and output the path for next round's use. If the result(s) contain(s) a lot of data, save the result(s) locally, output the path before proceed.
        - DO NOT USE `finish` tool before executing the code if code execution is required. If the result involves a lot of data, save the data as csv, txt, json file(s) etc. for the ease of processing in the following subtasks.
        """


def process_subtasks(
    subtasks: List[Dict[str, Any]],
    task: str,
    solved_dependent_sub_tasks: str,
) -> str:
    """
    Process and solve subtasks recursively while handling failures and replanning when necessary.
    This function implements a robust task-solving pipeline that includes verification,
    failure tracking, and dynamic replanning capabilities.

    Args:
        subtasks (List[Dict[str, Any]]): List of subtasks to be processed, where each subtask
            is a dictionary containing task instructions and metadata.
        task (str): The overall task description that provides context for subtask processing.
        solved_dependent_sub_tasks (str): String containing the accumulated results and context
            from previously solved subtasks.

    Returns:
        str: The final synthesized answer after processing all subtasks, incorporating
            the results from successful subtask executions and any necessary replanning.
    """
    global total_failure_count, max_total_failures, replanner_agent
    subtask_index: int = 0
    aggregated_result: str = ""
    while subtask_index < len(subtasks):
        print("current subtask:", subtasks[subtask_index])
        if subtask_index > 0:
            solved_dependent_sub_tasks += str(subtasks[subtask_index - 1])
        prompt: str = STRUCTURAL_PROMPT.format(
            overall_task=task,
            solved_sub_tasks=solved_dependent_sub_tasks,
            current_sub_task=subtasks[subtask_index],
        )
        msg: Msg = Msg(name="Planner", role="system", content=prompt)

        verdict: str = "non"
        failure_count: int = 0
        max_failure: int = 3  # Adjust as needed
        result: Optional[Msg] = None
        while "True" not in verdict[-5:]:
            if verdict != "non":
                msg = Msg(
                    name="Planner",
                    role="system",
                    content=prompt + " VERDICT: " + verdict,
                )
                failure_count += 1
                total_failure_count += 1

            if failure_count > max_failure:
                # Check if total failures exceed max_total_failures
                if total_failure_count > max_total_failures:
                    print("Exceeded maximum total failures. Aborting.")
                    return (
                        "Failed to solve the task due to excessive failures."
                    )

                # Call the replanner agent
                result_content: str = str(result.content) if result else ""
                msg_replan: Msg = Msg(
                    name="replanner",
                    role="system",
                    content=(
                        "overall_task: "
                        + task
                        + "\nsolved_dependent_sub_tasks: "
                        + solved_dependent_sub_tasks
                        + "\ncurrent_sub_task: "
                        + subtasks[subtask_index]["instruction"]
                        + "\nresult: "
                        + result_content
                        + "\nVERDICT: "
                        + verdict
                        + "\nall_subtasks: "
                        + str(subtasks)
                    ),
                )
                decision: str
                output: Any
                decision, output = replanner_agent(msg_replan)
                if decision == "decompose_subtask":
                    # Decompose current subtask into sub-subtasks
                    print("Decomposing current subtask into sub-subtasks...")
                    # Recursively process the new subtasks
                    final_answer = process_subtasks(
                        output,
                        task,
                        solved_dependent_sub_tasks,
                    )
                    # After processing sub-subtasks, set the result of current subtask
                    subtasks[subtask_index]["result"] = final_answer
                    aggregated_result += final_answer
                    subtask_index += 1
                    break  # Break the while loop
                if decision == "replan_subtask":
                    # Update subtasks with the new plan
                    print("Replanning current and subsequent subtasks...")
                    # Replace current and subsequent subtasks
                    # subtasks = subtasks[:subtask_index] + output
                    subtasks = copy.deepcopy(output)
                    # Reset failure_count
                    failure_count = 0
                    # Continue with the updated subtasks
                    break  # Break and restart the while loop with new subtasks
                raise ValueError(
                    "Unknown decision from replanning_agent.",
                )

            # Proceed with solving the subtask
            result = solver_agent(msg)
            msg_verifier: Msg = Msg(
                name="Verifier",
                role="system",
                content=(
                    "overall_task: "
                    + task
                    + "\nsolved_dependent_sub_tasks: "
                    + solved_dependent_sub_tasks
                    + "\ncurrent_sub_task: "
                    + subtasks[subtask_index]["instruction"]
                    + "\nresult: "
                    + str(result.content)
                ),
            )
            verdict = verifier_agent(msg_verifier).content

        # Store the result if verification passed
        if "True" in verdict[-5:]:
            subtasks[subtask_index]["result"] = str(result.content)
            aggregated_result += str(result.content)
            subtask_index += 1  # Move to the next subtask
            # Reset failure_count after a successful subtask
            failure_count = 0

    # Once all subtasks are processed, synthesize the final answer
    msg_synthesizer: Msg = Msg(
        name="synthesizer",
        role="system",
        content="overall_task: " + task + "\nsubtasks: " + str(subtasks),
    )
    final_answer = synthesizer_agent(msg_synthesizer).content
    return final_answer


def problem_solving(task: str) -> str:
    """
    Solve the given task by planning, processing subtasks, and synthesizing the final answer.

    Args:
        task (str): The task description to be solved.

    Returns:
        str: The final solution to the task.
    """
    global total_failure_count, max_total_failures
    total_failure_count = 0
    max_total_failures = 10  # Adjust as needed

    task_msg: Msg = Msg(name="Planner", role="system", content=task)
    subtasks: List[Dict[str, Any]] = planner_agent(task_msg)
    solved_dependent_sub_tasks: str = ""

    final_answer = process_subtasks(subtasks, task, solved_dependent_sub_tasks)

    return final_answer


def init_agents() -> None:
    """
    Initialize all agents with the required configurations.
    """
    global planner_agent, solver_agent, verifier_agent, synthesizer_agent, replanner_agent

    agentscope.init(
        model_configs=[
            {
                "config_name": "gpt_config",
                "model_type": "openai_chat",
                # "model_name": "chatgpt-4o-latest",
                "model_name": "gpt-4o-mini",
                # "model_name": "gpt-4o",
                # "model_name": "o1-mini",
                "api_key": openai_api_key,
                "generate_args": {
                    "temperature": 0.0,
                },
            },
            {
                "config_name": "lite_llm_claude",
                "model_type": "litellm_chat",
                "model_name": "claude-3-5-haiku-20241022",
                # "model_name": "claude-3-5-sonnet-20241022",
                "generate_args": {
                    # "max_tokens": 4096,
                    "temperature": 0.0,
                },
            },
            {
                "model_type": "post_api_chat",
                "config_name": "my_post_api",
                "api_url": "https://xxx",
                "headers": {},
            },
            {
                "config_name": "dashscope_chat",
                "model_type": "dashscope_chat",
                "model_name": "qwen-plus",
                "api_key": dashscope_api_key,
                "generate_args": {
                    "temperature": 0.0,
                },
            },
        ],
        project="Multi-Agent Conversation",
        save_api_invoke=True,
        use_monitor=True,  # Enable token usage monitoring
    )

    # Create a ServiceToolkit instance
    service_toolkit = ServiceToolkit()
    # Add your tools to the service_toolkit here if needed
    service_toolkit.add(
        execute_python_code,
    )
    service_toolkit.add(
        list_directory_content,
    )
    service_toolkit.add(
        get_current_directory,
    )
    service_toolkit.add(
        execute_shell_command,
    )

    # Initialize the agents
    planner_agent = PlannerAgent(
        name="planner",
        sys_prompt="You're a helpful assistant.",
        model_config_name="dashscope_chat",
        service_toolkit=service_toolkit,
    )

    solver_agent = ReActAgent(
        name="solver",
        sys_prompt="You're a helpful assistant.",
        model_config_name="dashscope_chat",
        service_toolkit=service_toolkit,
    )

    # Overwrite the parser attribute with the custom format_instruction to reinforce the output adhere to json format.
    solver_agent.parser = RegexTaggedContentParser(
        format_instruction="""Respond with specific tags as outlined below in json format:
        <thought>{what you thought}</thought>
        <function>{the function name you want to call}</function>
        <{argument name}>{argument value}</{argument name}>
        <{argument name}>{argument value}</{argument name}>
        ...""",  # noqa
        try_parse_json=True,
        required_keys=["thought", "function"],
    )

    verifier_agent = VerifierAgent(
        name="verifier",
        sys_prompt="You're a helpful assistant.",
        model_config_name="dashscope_chat",
        service_toolkit=service_toolkit,
    )

    synthesizer_agent = SynthesizerAgent(
        name="synthesizer",
        sys_prompt="You're a helpful assistant.",
        model_config_name="dashscope_chat",
    )

    replanner_agent = ReplanningAgent(
        name="replanner",
        sys_prompt="You're a helpful assistant.",
        model_config_name="dashscope_chat",
        service_toolkit=service_toolkit,
    )


def main() -> None:
    """Initialize agents and run an example task through the problem-solving pipeline."""
    # Initialize agents
    init_agents()

    # Example task (you can replace this with any task)
    input_task = "Your task description here."
    final_solution = problem_solving(input_task)
    print("final solution: ", final_solution)


if __name__ == "__main__":
    main()
