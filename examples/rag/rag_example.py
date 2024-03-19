# -*- coding: utf-8 -*-
"""
A simple example for conversation between user and
an agent with RAG capability.
"""
import os
import argparse
from rag_agents import LlamaIndexAgent, LangChainRAGAgent
import agentscope
from agentscope.agents import UserAgent


def main() -> None:
    """A conversation demo"""

    agentscope.init(
        model_configs=[
            {
                "model_type": "dashscope_chat",
                "config_name": "qwen_config",
                "model_name": "qwen-max",
                "api_key": f"{os.environ.get('DASHSCOPE_API_KEY')}",
            },
            {
                "model_type": "dashscope_text_embedding",
                "config_name": "qwen_emb_config",
                "model_name": "text-embedding-v2",
                "api_key": f"{os.environ.get('DASHSCOPE_API_KEY')}",
            },
        ],
    )

    # Init RAG agent and user
    if args.module == "llamaindex":
        rag_agent = LlamaIndexAgent(
            name="Assistant",
            sys_prompt="You're a helpful assistant. You need to generate "
            "answers based on the provided context:\n "
            "Context: \n {retrieved_context}\n ",
            model_config_name="qwen_config",  # model config name
            emb_model_config_name="qwen_emb_config",
            config={"data_path": args.data_path},
        )
    else:
        rag_agent = LangChainRAGAgent(
            name="Assistant",
            sys_prompt="You're a helpful assistant. You need to generate"
            " answers based on the provided context:\n "
            "Context: \n {retrieved_context}\n ",
            model_config_name="qwen_config",  # your model config name
            emb_model_config_name="qwen_emb_config",
            config={"data_path": args.data_path},
        )
    user_agent = UserAgent()
    # start the conversation between user and assistant
    while True:
        x = user_agent()
        x.role = "user"  # to enforce dashscope requirement on roles
        if len(x["content"]) == 0 or str(x["content"]).startswith("exit"):
            break
        rag_agent(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        choices=["llamaindex", "langchain"],
        default="llamaindex",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/",
    )
    args = parser.parse_args()
    main()
