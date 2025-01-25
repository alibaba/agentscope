# -*- coding: utf-8 -*-
"""conversation between user and stable-diffusion agent."""
import agentscope
from agentscope.agents import DialogAgent
from agentscope.agents.user_agent import UserAgent


def main() -> None:
    """A basic conversation demo"""

    agentscope.init(
        model_configs=[
            {
                "model_type": "sd_txt2img",
                "config_name": "sd",
                "options": {
                    "sd_model_checkpoint": "xxxxxx",
                    "CLIP_stop_at_last_layers": 2,
                },
                "generate_args": {
                    "steps": 50,
                    "n_iter": 1,
                },
            },
        ],
        project="txt2img-Agent Conversation",
        save_api_invoke=True,
    )

    # Init two agents
    dialog_agent = DialogAgent(
        name="Assistant",
        sys_prompt="dreamy",  # replace by your image style prompts
        model_config_name="sd",  # replace by your model config name
    )
    user_agent = UserAgent()

    # start the conversation between user and assistant
    msg = None
    while True:
        msg = user_agent(msg)
        if msg.content == "exit":
            break
        msg = dialog_agent(msg)


if __name__ == "__main__":
    main()
