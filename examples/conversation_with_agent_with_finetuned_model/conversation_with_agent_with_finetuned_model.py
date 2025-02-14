# -*- coding: utf-8 -*-
"""
This script sets up a conversational agent using
AgentScope with a Hugging Face model.
It includes initializing a FinetuneDialogAgent,
loading and fine-tuning a pre-trained model,
and conducting a dialogue via a sequential pipeline.
The conversation continues until the user exits.
Features include model and tokenizer loading,
and fine-tuning on the lima dataset with adjustable parameters.
"""
# This import is necessary for AgentScope to properly use
# HuggingFaceWrapper even though it's not explicitly used in this file.
# To remove the pylint disable without causing issues
# HuggingFaceWrapper needs to be put under src/agentscope/agents.
# pylint: disable=unused-import
from huggingface_model import HuggingFaceWrapper
from FinetuneDialogAgent import FinetuneDialogAgent
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
                "pretrained_model_name_or_path": "google/gemma-7b",
                # "local_model_path": "", # Specify your local model path
                "max_length": 256,
                # Device for inference. Fine-tuning occurs on gpus.
                "device": "cuda",
                # Specify a Hugging Face data path if you
                # wish to finetune the model from the start
                "data_path": "GAIR/lima",
                # "output_dir":
                # fine_tune_config (Optional): Configuration for
                # fine-tuning the model.
                # This dictionary can include hyperparameters and other
                # training options that will be passed to the
                # fine-tuning method. Defaults to None.
                # `lora_config` and `training_args` follow
                # the standard lora and sfttrainer fields.
                # "lora_config" shouldn't be specified if
                # loading a model saved as lora model
                # '"continue_lora_finetuning": True' if
                # loading a model saved as lora model
                "fine_tune_config": {
                    "continue_lora_finetuning": False,
                    "max_seq_length": 4096,
                    "lora_config": {
                        "r": 16,
                        "lora_alpha": 32,
                        "lora_dropout": 0.05,
                        "bias": "none",
                        "task_type": "CAUSAL_LM",
                    },
                    "training_args": {
                        "num_train_epochs": 5,
                        # "max_steps": 100,
                        "logging_steps": 1,
                        # "learning_rate": 5e-07
                    },
                    # "bnb_config": {
                    #     "load_in_8bit": True,
                    # "bnb_4bit_use_double_quant": True,
                    # "bnb_4bit_quant_type": "nf4",
                    # "bnb_4bit_compute_dtype": "bfloat16",
                    # },
                },
            },
        ],
    )

    # # alternatively can load `model_configs` from json file
    # agentscope.init(
    #     model_configs="./configs/model_configs.json",
    # )

    # Init agents with the custom model
    dialog_agent = FinetuneDialogAgent(
        name="Assistant",
        sys_prompt=("You're a helpful assistant."),
        # Use your custom model config name here
        model_config_name="my_custom_model",
    )

    # (Optional) can load another model after
    # the agent has been instantiated if needed
    # (for `fine_tune_config` specify only
    # `lora_config` and `bnb_config` if used)
    dialog_agent.load_model(
        pretrained_model_name_or_path="google/gemma-7b",
        # local_model_path="",
        fine_tune_config={
            # "bnb_config": {
            #     "load_in_4bit": True,
            #     "bnb_4bit_use_double_quant": True,
            #     "bnb_4bit_quant_type": "nf4",
            #     "bnb_4bit_compute_dtype": "bfloat16",
            # },
        },
    )  # load model from Hugging Face

    dialog_agent.load_tokenizer(
        pretrained_model_name_or_path="google/gemma-7b",
        # local_model_path="",
    )  # load tokenizer

    # fine-tune loaded model with lima dataset
    # with customized hyperparameters
    # `fine_tune_config` argument is optional
    # specify only `lora_config` and
    # `training_args` if used). Defaults to None.
    # "lora_config" shouldn't be specified if
    # loading a model saved as lora model
    # '"continue_lora_finetuning": True' if
    # loading a model saved as lora model
    dialog_agent.fine_tune(
        "GAIR/lima",
        fine_tune_config={
            "continue_lora_finetuning": True,
            # "lora_config": {"r": 24, "lora_alpha": 48},
            "training_args": {"max_steps": 300, "logging_steps": 3},
        },
    )

    user_agent = UserAgent()

    # Start the conversation between user and assistant
    x = None
    while x is None or x.content != "exit":
        x = sequentialpipeline([user_agent, dialog_agent], x)


if __name__ == "__main__":
    main()
