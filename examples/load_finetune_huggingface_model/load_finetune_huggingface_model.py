# -*- coding: utf-8 -*-
"""
This script sets up a conversational agent using
AgentScope with a Hugging Face model.
It includes initializing a Finetune_DialogAgent,
loading and fine-tuning a pre-trained model,
and conducting a dialogue via a sequential pipeline.
The conversation continues until the user exits.
Features include model and tokenizer loading,
and fine-tuning on the GAIR/lima dataset with adjustable parameters.
"""
from huggingface_model import Finetune_DialogAgent

import agentscope
from agentscope.agents.user_agent import UserAgent
from agentscope.pipelines.functional import sequentialpipeline


def main() -> None:
    """A basic conversation demo with a custom model"""

    # Initialize AgentScope with your custom model configuration

    agentscope.init(
        model_configs=[
            {
                "model_type": "huggingface",
                "config_name": "my_custom_model",
                # Or another generative model of your choice.
                # Needed from loading from Hugging Face.
                "model_id": "google/gemma-2b-it",
                # "local_model_path":  # Specify your local model path
                # "local_tokenizer_path":  # Specify your local tokenizer path
                "max_length": 128,
                "device": "cuda",
                # Specify a Hugging Face data path if you
                # wish to finetune the model from the start
                "data_path": "GAIR/lima",
                # fine_tune_config (Optional): Configuration for
                # fine-tuning the model.
                # This dictionary can include hyperparameters and other
                # training options that will be passed to the
                # fine-tuning method. Defaults to None.
                "fine_tune_config": {
                    "lora_config": {"r": 20, "lora_alpha": 40},
                    "training_args": {"max_steps": 20, "logging_steps": 2},
                },
            },
        ],
    )

    # Init agents with the custom model
    dialog_agent = Finetune_DialogAgent(
        name="Assistant",
        sys_prompt="You're a helpful assistant.",
        # Use your custom model config name here
        model_config_name="my_custom_model",
    )

    dialog_agent.load_model(
        model_id="google/gemma-2b-it",
        local_model_path=None,
    )  # load model gemma-2b-it from Hugging Face
    dialog_agent.load_tokenizer(
        model_id="google/gemma-2b-it",
        local_tokenizer_path=None,
    )  # load tokenizer for gemma-2b-it from Hugging Face

    # fine-tune loaded model with lima dataset with default hyperparameters
    # dialog_agent.fine_tune(data_path=  "GAIR/lima")

    # fine-tune loaded model with lima dataset with customized hyperparameters
    # (`fine_tune_config` argument is optional. Defaults to None.)
    dialog_agent.fine_tune(
        "GAIR/lima",
        fine_tune_config={
            "lora_config": {"r": 24, "lora_alpha": 48},
            "training_args": {"max_steps": 30, "logging_steps": 3},
        },
    )

    user_agent = UserAgent()

    # Start the conversation between user and assistant
    x = None
    while x is None or x.content != "exit":
        x = sequentialpipeline([dialog_agent, user_agent], x)


if __name__ == "__main__":
    main()
