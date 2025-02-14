# User-Agent Conversation with Custom Model Loading and Fine-Tuning in AgentScope

This example demonstrates how to load and optionally fine-tune a Hugging Face model within a user-agent conversation setup using AgentScope. The complete code is provided in `agentscope/examples/conversation_with_agent_with_finetuned_model`.

## Functionality Overview

Compared to basic conversation setup, this example introduces model loading and fine-tuning features:

- Initialize an agent or use `dialog_agent.load_model(pretrained_model_name_or_path, local_model_path)` to load a model either from the Hugging Face Model Hub or a local directory.
- Initalize an agent or apply `dialog_agent.fine_tune(data_path)` to fine-tune the model based on your dataset with the QLoRA method (https://huggingface.co/blog/4bit-transformers-bitsandbytes).

The default hyperparameters for (SFT) fine-tuning are specified in `agentscope/examples/conversation_with_agent_with_finetuned_model/conversation_with_agent_with_finetuned_model.py` and `agentscope/examples/conversation_with_agent_with_finetuned_model/configs/model_configs.json`. For customized hyperparameters, specify them in `model_configs` if the model needs to be fine-tuned at initialization, or specify through `fine_tune_config` in `FinetuneDialogAgent`'s `fine_tune` method after initialization, as shown in the example script `conversation_with_agent_with_finetuned_model.py`.

## Agent Initialization

When initializing an agent, the following parameters need specification:

- `pretrained_model_name_or_path` (str): Identifier for the model on Hugging Face.
- `local_model_path` (str): Local path to the model (defaults to loading from Hugging Face if not provided).
- `data_path` (str): Path to training data (fine-tuning is skipped if not provided).
- `device` (str): The device (e.g., 'cuda', 'cpu') for model operation, defaulting to 'cuda' if available.
- `fine_tune_config` (dict, Optional): A configuration dictionary for fine-tuning the model. It allows specifying hyperparameters and other training options that will be passed to the fine-tuning method. If not provided, default settings will be used. This allows for customization of the fine-tuning process to optimize model performance based on specific requirements. Note that if `continue_lora_finetuning` is set to `True`, `lora_config` should not be specified since the previously saved peft model's config will be used instead. If `continue_lora_finetuning` is set to `False`, `lora_config` should be specified.
- `huggingface_token` (from .env file): Token required for models needing authentication from Hugging Face.

## Example Ouputs
### Before Fine-tuning:
```
System:

You're a helpful assistant.

User:

Who are you?

Assistant:

I am a woman who is passionate about life, my family, my friends, my work, and my community. I am a woman who is a wife, a mother, a daughter, a sister, a friend, a teacher, a coach, a mentor, a leader, a volunteer, a writer, a reader, a traveler, a gardener, a cook, a baker, a cra
```
### After Fine-tuning (with the default configuration in `model_configs`):
```
System:

You're a helpful assistant.

User:

Who are you?

Assistant:

I am a language model trained by Google to answer questions.
```
(This example is trained with the default setting, with training time 872 seconds and 9.914 GB gpu memory cost. Reduce training batch size can reduce the memory required. Note that the model is loaded in 4 bits (i.e., QLoRA)).

## Tested Models

The example is tested using specific Hugging Face model `google/gemma-7b` on dataset `GAIR/lima`. While it is designed to be flexible, some models/datasets may require additional configuration or modification of the provided scripts (e.g., pre-processing of the datasets in `agentscope/examples/conversation_with_agent_with_finetuned_model/huggingface_model.py`).

## Prerequisites

Before running this example, ensure you have installed the following packages:

- `transformers`
- `python-dotenv`
- `datasets`
- `trl`
- `bitsandbytes`
- `sentencepiece`

Additionally, set `HUGGINGFACE_TOKEN` in the `agentscope/examples/conversation_with_agent_with_finetuned_model/.env`.

```bash
python conversation_with_agent_with_finetuned_model.py