# -*- coding: utf-8 -*-
"""
An example for conversation between user and agents with RAG capability.
One agent is a tutorial assistant, the other is a code explainer.
"""
import json
import os

from groupchat_utils import filter_agents

import agentscope
from agentscope.agents import UserAgent
from agentscope.rag import KnowledgeBank


AGENT_CHOICE_PROMPT = """
There are following available agents. You need to choose the most appropriate
agent(s) to answer the user's question.

agent descriptions:{}

First, rephrase the user's question, which must contain the key information.
The you need to think step by step. If you believe some of the agents are
good candidates to answer the question (e.g., AGENT_1 and AGENT_2), then
you need to follow the following format to generate your output:

'
Because $YOUR_REASONING.
I believe @AGENT_1 and @AGENT_2 are the most appropriate agents to answer
your question.
'
"""


def prepare_docstring_html() -> None:
    """prepare docstring in html for API assistant"""
    if not os.path.exists("../../docs/docstring_html/"):
        os.system(
            "sphinx-apidoc -f -o ../../docs/sphinx_doc/en/source "
            "../../src/agentscope -t template",
        )
        os.system(
            "sphinx-build -b html  ../../docs/sphinx_doc/en/source "
            "../../docs/docstring_html/ -W --keep-going",
        )


def main() -> None:
    """A RAG multi-agent demo"""
    # prepare html for api agent
    prepare_docstring_html()

    # prepare models
    with open("configs/model_config.json", "r", encoding="utf-8") as f:
        model_configs = json.load(f)
    # for internal API
    for config in model_configs:
        if config.get("model_type", "") == "post_api_chat":
            config["headers"]["Authorization"] = (
                "Bearer " + f"{os.environ.get('HTTP_LLM_API_KEY')}"
            )
        else:
            # for dashscope
            config["api_key"] = f"{os.environ.get('DASHSCOPE_API_KEY')}"

    # load config of the agents
    with open("configs/agent_config.json", "r", encoding="utf-8") as f:
        agent_configs = json.load(f)

    agent_list = agentscope.init(
        model_configs=model_configs,
        agent_configs=agent_configs,
    )
    rag_agent_list = agent_list[:4]
    guide_agent = agent_list[4]

    # the knowledge bank can be configured by loading config file
    with open(
        "configs/knowledge_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        knowledge_configs = json.load(f)
    knowledge_bank = KnowledgeBank(configs=knowledge_configs)

    # alternatively, we can easily input the configs to add data to RAG
    knowledge_bank.add_data_as_knowledge(
        knowledge_id="agentscope_tutorial_rag",
        emb_model_name="qwen_emb_config",
        data_dirs_and_types={
            "../../docs/sphinx_doc/en/source/tutorial": [".md"],
        },
    )

    # let knowledgebank to equip rag agent with a (set of) knowledge
    for agent in rag_agent_list:
        knowledge_bank.equip(agent)

    rag_agent_names = [agent.name for agent in rag_agent_list]

    # update guide agent system prompt with the descriptions of rag agents
    rag_agent_descriptions = [
        "agent name: "
        + agent.name
        + "\n agent description："
        + agent.description
        + "\n"
        for agent in rag_agent_list
    ]

    guide_agent.sys_prompt = (
        guide_agent.sys_prompt
        + AGENT_CHOICE_PROMPT.format(
            "".join(rag_agent_descriptions),
        )
    )

    user_agent = UserAgent()
    while True:
        # The workflow is the following:
        # 1. user input a message,
        # 2. if it mentions one of the agents, then the agent will be called
        # 3. otherwise, the guide agent will decide which agent to call
        # 4. the called agent will respond to the user
        # 5. repeat
        x = user_agent()
        x.role = "user"  # to enforce dashscope requirement on roles
        if len(x["content"]) == 0 or str(x["content"]).startswith("exit"):
            break
        speak_list = filter_agents(x.get("content", ""), rag_agent_list)
        if len(speak_list) == 0:
            guide_response = guide_agent(x)
            # Only one agent can be called in the current version,
            # we may support multi-agent conversation later
            speak_list = filter_agents(
                guide_response.get("content", ""),
                rag_agent_list,
            )
        agent_name_list = [agent.name for agent in speak_list]
        for agent_name, agent in zip(agent_name_list, speak_list):
            if agent_name in rag_agent_names:
                agent(x)


if __name__ == "__main__":
    main()
