# -*- coding: utf-8 -*-
"""
This module provides a HuggingFaceWrapper to manage
and operate Hugging Face Transformers models, enabling loading,
fine-tuning, and response generation.
Key features include handling model and tokenizer operations,
adapting to specialized datasets, and robust error management.
"""
from typing import Sequence, Any, Union, List, Optional, Dict, Tuple
import os
from datetime import datetime
import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import transformers
from peft import LoraConfig
from peft import get_peft_model
from peft import PeftModel, PeftConfig
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from loguru import logger
from dotenv import load_dotenv

from agentscope.models import ModelWrapperBase, ModelResponse
from agentscope.message import MessageBase
from agentscope.utils.tools import _convert_to_str


class HuggingFaceWrapper(ModelWrapperBase):
    """Wrapper for a Hugging Face transformer model.

    This class is responsible for loading and fine-tuning
    pre-trained models from the Hugging Face library.
    """

    model_type: str = "huggingface"  # Unique identifier for this model wrapper

    def __init__(
        self,
        config_name: str,
        pretrained_model_name_or_path: Optional[str] = None,
        max_length: int = 512,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        local_model_path: Optional[str] = None,
        fine_tune_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the HuggingFaceWrapper with the given configuration.

        Arguments:
            config_name (str): Configuration name for model setup.
            pretrained_model_name_or_path (str): Identifier for
                                        the pre-trained model on
                                        Hugging Face.
            max_length (int): Maximum sequence length for the
                              model output per reply.
                              Defaults to 512.
            data_path (str, optional): Path to the dataset for
                                       fine-tuning the model.
            output_dir (str, optional): User specified path to save
                                        the fine-tuned model
                                        and its tokenizer. By default
                                        save to this example's
                                        directory if not specified.
            device (torch.device, optional): Device to run the model on.
                                             Default to GPU if available.
            local_model_path (str, optional): Local file path to a
                                              pre-trained model
                                              and its tokenizer.
            fine_tune_config (dict, optional): Configuration for
                                               fine-tuning the model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            config_name=config_name,
            model_name=pretrained_model_name_or_path,
        )
        self.model = None
        self.tokenizer = None
        self.max_length = max_length
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.local_model_path = local_model_path
        self.lora_config = None
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        dotenv_path = os.path.join(script_dir, ".env")
        _ = load_dotenv(dotenv_path)  # read local .env file
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        if device is None:
            self.device_map = "auto"
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self.device_map = device
            self.device = device

        self.load_model(
            pretrained_model_name_or_path,
            local_model_path=local_model_path,
            fine_tune_config=fine_tune_config,
        )
        self.load_tokenizer(
            pretrained_model_name_or_path,
            local_model_path=local_model_path,
        )

        if data_path is not None:
            self.model = self.fine_tune_training(
                self.model,
                self.tokenizer,
                data_path,
                output_dir,
                token=self.huggingface_token,
                fine_tune_config=fine_tune_config,
            )

    def __call__(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Process the input data to generate a response from the model.

        This method tokenizes the input text, generates
        a response using the model,
        and then decodes the generated tokens into a string.

        Arguments:
            input (list): A list of dictionaries where each dictionary contains
                          'name', 'role' and 'content' keys
                          and their respective values.
            **kwargs: Additional keyword arguments for the
                      model's generate function.

        Returns:
            ModelResponse: An object containing the generated
                           text and raw model output.

        Raises:
            Exception: If an error occurs during text generation.
        """
        if self.tokenizer is not None:
            self.model, self.tokenizer = self._setup_model_and_tokenizer(
                self.model,
                self.tokenizer,
            )
        else:
            logger.error("Tokenizer is not initialized")

        try:
            input_ids = self.tokenizer.apply_chat_template(
                inputs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # Generate response using the model
            outputs = self.model.generate(
                input_ids.to(self.device),
                max_new_tokens=self.max_length,
                **kwargs,
            )
            # Decode the generated tokens to a string
            generated_text = self.tokenizer.decode(
                outputs[0][input_ids.shape[1] :],  # noqa: E203
                skip_special_tokens=True,
            )
            return ModelResponse(text=generated_text, raw=outputs)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def format(
        self,
        *args: Union[MessageBase, Sequence[MessageBase]],
    ) -> List[dict]:
        """A basic strategy to format the input into the required format of
        Hugging Face models.

        Args:
            args (`Union[MessageBase, Sequence[MessageBase]]`):
                The input arguments to be formatted, where each argument
                should be a `Msg` object, or a list of `Msg` objects.
                In distribution, placeholder is also allowed.

        Returns:
            `List[dict]`:
                The formatted messages.
        """
        huggingface_msgs = []
        for msg in args:
            if msg is None:
                continue
            if isinstance(msg, MessageBase):
                # content shouldn't be empty string
                if msg.content == "":
                    logger.warning(
                        "The content field cannot be "
                        "empty string. To avoid error, the empty string is "
                        "replaced by a blank space automatically, but the "
                        "model may not work as expected.",
                    )
                    msg.content = " "

                huggingface_msg = {
                    "role": msg.role,
                    "content": _convert_to_str(msg.content),
                }

                # image url
                if msg.url is not None:
                    huggingface_msg["images"] = [msg.url]

                huggingface_msgs.append(huggingface_msg)
            elif isinstance(msg, list):
                huggingface_msgs.extend(self.format(*msg))
            else:
                raise TypeError(
                    f"Invalid message type: {type(msg)}, `Msg` is expected.",
                )

        return huggingface_msgs

    def load_model(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        local_model_path: Optional[str] = None,
        fine_tune_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Load a new model for the agent from
        a local path and update the agent's model.

        Arguments:
            local_model_path (str): The file path to the model to be loaded.
            pretrained_model_name_or_path (str): An identifier for
                                                 the model on Huggingface.

        Raises:
            Exception: If the model cannot be loaded from the given
                       path or identifier.
                       Possible reasons include file not found,
                       incorrect model ID,
                       or network issues while fetching the model.
        """
        self.local_model_path = local_model_path
        bnb_config = None
        bnb_config_default = {}

        if fine_tune_config is not None:
            if fine_tune_config.get("bnb_config") is not None:
                bnb_config_default.update(fine_tune_config["bnb_config"])
        if bnb_config_default:
            bnb_config = BitsAndBytesConfig(**bnb_config_default)

        try:
            if local_model_path is None:
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    device_map=self.device_map,
                    torch_dtype=torch.bfloat16,
                    **(
                        {"quantization_config": bnb_config}
                        if bnb_config is not None
                        else {}
                    ),
                    token=self.huggingface_token,
                )

                info_msg = (
                    f"Successfully loaded new model "
                    f"'{pretrained_model_name_or_path}' from "
                    f"Hugging Face"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    device_map=self.device_map,
                    torch_dtype=torch.bfloat16,
                    **(
                        {"quantization_config": bnb_config}
                        if bnb_config is not None
                        else {}
                    ),
                    local_files_only=True,
                )

                info_msg = (
                    f"Successfully loaded new model "
                    f"'{pretrained_model_name_or_path}' from "
                    f"'{local_model_path}'"
                )

            # log the successful model loading
            logger.info(info_msg)

        except Exception as e:
            # Handle exceptions during model loading,
            # such as file not found or load errors
            error_msg = (
                f"Failed to load model '{pretrained_model_name_or_path}' "
                f"from '{local_model_path}': {e}"
            )

            logger.error(error_msg)

            raise

    def load_tokenizer(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        local_model_path: Optional[str] = None,
    ) -> None:
        """
        Load the tokenizer from a local path.

        Arguments:
            local_model_path (str): The file path to the
                                    tokenizer to be loaded
                                    (same as `local_model_path`).
            pretrained_model_name_or_path (str): An identifier
                                                for the model on Huggingface.

        Raises:
            Exception: If the tokenizer cannot be loaded from the
            given path or identifier. Possible reasons include file not found,
            incorrect model ID, or network issues while fetching the tokenizer.
        """
        self.local_model_path = local_model_path
        try:
            if local_model_path is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path,
                    token=self.huggingface_token,
                )
                # log the successful tokenizer loading
                logger.info(
                    f"Successfully loaded new tokenizer for model "
                    f"'{pretrained_model_name_or_path}' from Hugging Face",
                )

            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_model_path,
                )
                # log the successful tokenizer loading
                logger.info(
                    f"Successfully loaded new tokenizer for model "
                    f"'{pretrained_model_name_or_path}'"
                    f" from '{local_model_path}'",
                )
            self.tokenizer.padding_side = "right"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            # Handle exceptions during model loading,
            # such as file not found or load errors
            error_message = (
                f"Failed to load tokenizer for model"
                f" '{pretrained_model_name_or_path}' from "
                f"'{local_model_path}': {e}"
            )
            logger.error(error_message)

            raise

    def setup_chat_format(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        resize_to_multiple_of: Optional[int] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Setup chat format by adding special tokens to the tokenizer,
        setting the correct format, and extending the embedding layer
        of the model based on the new special tokens.

        Args:
        model (`~transformers.PreTrainedModel`): The model to be modified.
        tokenizer (`~transformers.PreTrainedTokenizer`): The tokenizer
                                                         to be modified.
        format (`Optional[Literal["chatml"]]`): The format to be set.
                                                Defaults to "chatml".
        resize_to_multiple_of (`Optional[int]`): Number to resize
                                                 the embedding layer to.
                                                 Defaults to None.
        Returns:
        model (`~transformers.PreTrainedModel`): modified model.
        tokenizer (`~transformers.PreTrainedTokenizer`): modified tokenizer.
        """

        # set special tokens and them
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|system|>",
                    "<|user|>",
                    "<|assistant|>",
                ],
            },
        )
        # set chat format for tokenizer
        tokenizer.chat_template = """{% for message in messages %}
        {% if message['role'] == 'user' %}
        {{ '<|user|>\n' + message['content'] + eos_token }}
        {% elif message['role'] == 'system' %}
        {{ '<|system|>\n' + message['content'] + eos_token }}
        {% elif message['role'] == 'assistant' %}
        {{ '<|assistant|>\n'  + message['content'] + eos_token }}
        {% endif %}
        {% if loop.last and add_generation_prompt %}
        {{ '<|assistant|>' }}
        {% endif %}
        {% endfor %}"""

        # resize embedding layer to a multiple of 64,
        # https://x.com/karpathy/status/1621578354024677377
        model.resize_token_embeddings(
            len(tokenizer),
            pad_to_multiple_of=resize_to_multiple_of
            if resize_to_multiple_of is not None
            else None,
        )
        # Update the model config to use the new eos & bos tokens
        if getattr(model, "config", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.bos_token_id = tokenizer.bos_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
        # Update the generation config to use the new eos & bos token
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.bos_token_id = tokenizer.bos_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    def _setup_model_and_tokenizer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Set up the model and tokenizer based on the
        local model path and chat template.

        This method checks if a local model path exists
        and if the tokenizer has a chat template.
        It then sets up the chat format if necessary
        and loads pre-trained token embeddings.

        Args:
            model (PreTrainedModel): The pre-trained model to set up.
            tokenizer (PreTrainedTokenizer): The tokenizer
                                             associated with the model.

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]:
            The potentially modified model and tokenizer.

        Note:
            This method modifies the model and tokenizer
            in-place but also returns them for convenience.
        """
        if self.local_model_path is not None:
            if tokenizer.chat_template is not None:
                if model.get_input_embeddings().weight.size()[0] != len(
                    tokenizer,
                ):
                    model, tokenizer = self.setup_chat_format(model, tokenizer)
                    loaded_embed_tokens_weights = torch.load(
                        f"{self.local_model_path}/embed_tokens_weights.pt",
                    )
                    model.get_input_embeddings().weight.data.copy_(
                        loaded_embed_tokens_weights,
                    )
                    del loaded_embed_tokens_weights
                    torch.cuda.empty_cache()
            elif tokenizer.chat_template is None:
                model, tokenizer = self.setup_chat_format(model, tokenizer)
        elif tokenizer.chat_template is None:
            model, tokenizer = self.setup_chat_format(model, tokenizer)

        return model, tokenizer

    def fine_tune(
        self,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        fine_tune_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Fine-tune the agent's model using data from the specified path.

        Arguments:
            data_path (str): The file path to the training
                             data from Hugging Face.
            output_dir (str, optional): User specified path
                                       to save the fine-tuned model
                                       and its tokenizer. By default
                                       save to this example's
                                       directory if not specified.

        Raises:
            Exception: If the fine-tuning process fails. This could be
            due to issues with the data path, configuration parameters,
            or internal errors during the training process.
        """
        try:
            self.model = self.fine_tune_training(
                self.model,
                self.tokenizer,
                data_path,
                output_dir,
                token=self.huggingface_token,
                fine_tune_config=fine_tune_config,
            )

            logger.info(
                f"Successfully fine-tuned model with data from '{data_path}'",
            )
        except Exception as e:
            logger.error(
                f"Failed to fine-tune model with data from '{data_path}': {e}",
            )
            raise

    # Function to reformat a single row
    def _reformat_row(self, row: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """
        Reformat a single row of conversation data
        into a list of message dictionaries.

        This method takes a row from the dataset,
        which contains a list of conversation
        turns, and reformats it into a list of dictionaries.
        Each dictionary represents
        a message in the conversation, with 'role' and 'content' keys.

        Args:
            row (Dict[str, List[str]]): A dictionary containing a
                                        'conversations' key
                                        with a list of two strings:
                                        the user's input
                                        and the assistant's response.

        Returns:
            List[Dict[str, str]]: A list of three dictionaries
                                  representing the system message,
                                  user message, and assistant message.

        Example:
            Input row: {"conversations": ["User input", "Assistant response"]}
            Output: [
                {"role": "system", "content": "You're a helpful assistant."},
                {"role": "user", "content": "User input"},
                {"role": "assistant", "content": "Assistant response"}
            ]
        """
        return [
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": row["conversations"][0]},
            {"role": "assistant", "content": row["conversations"][1]},
        ]

    def _formatting_prompts_func(
        self,
        example: Dict[str, List[List[str]]],
    ) -> List[str]:
        """
        Formats each conversation in the dataset for training.
        Args:
            example (Dict[str, List[List[str]]]): A dataset.

        Returns:
            List[str]: A dataset with combined field.
        """
        output_texts = []
        for i in range(len(example["conversations"])):
            text = (
                "### Question: "
                + example["conversations"][i][0]
                + "\n ### Answer: "
                + example["conversations"][i][1]
            )
            output_texts.append(text)
        return output_texts

    def fine_tune_training(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        token: Optional[str] = None,
        fine_tune_config: Optional[Dict[str, Any]] = None,
    ) -> AutoModelForCausalLM:
        """
        The actual method that handles training and fine-tuning
        the model on the dataset specified by
        the data_path using a given tokenizer.

        Arguments:
            model (AutoModelForCausalLM): The pre-trained causal language model
                                          from Hugging Face's transformers.
            tokenizer (AutoTokenizer): The tokenizer corresponding to
                                       the pre-trained model.
            data_path (str): The file path or dataset identifier to load
                             the dataset from Hugging Face.
            output_dir (str, optional): User specified path
                                       to save the fine-tuned model
                                       and its tokenizer. By default
                                       save to this example's
                                       directory if not specified.
            token (str): The authentication token for Hugging Face.
            fine_tune_config (dict, optional): Configuration options for
                                               fine-tuning the model,
                                               including QLoRA and training
                                               arguments.

        Returns:
            AutoModelForCausalLM: The fine-tuned language model.

        Raises:
            Exception: Raises an exception if the dataset
                       loading or fine-tuning process fails.

        Note:
            This method updates the model in place and also logs
            the fine-tuning process.
            It utilizes the QLoRA configuration and custom training arguments
            to adapt the pre-trained model to the specific dataset.
            The training log and trained model are saved in the same
            directory with the specific timestamp at saving time
            as part of the log/model fodler name.
        """

        model, tokenizer = self._setup_model_and_tokenizer(model, tokenizer)

        dataset = load_dataset(data_path, split="train", token=token)

        # # filter out input sequences that are longer than certain threshold
        dataset_reduced = dataset.filter(
            lambda x: len(x["conversations"][0] + x["conversations"][1])
            <= 3000,
        )

        # Apply the reformatting function to each row in the dataset
        formatted_dataset = [
            self._reformat_row(row) for row in dataset_reduced
        ]
        formatted_dataset = Dataset.from_dict({"messages": formatted_dataset})

        training_defaults = {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "num_train_epochs": 1,
            "output_dir": "./",
            "optim": "paged_adamw_8bit",
            "logging_steps": 1,
        }
        max_seq_length_default = 4096

        lora_config_default = {}

        if fine_tune_config is not None:
            if fine_tune_config.get("max_seq_length") is not None:
                max_seq_length_default = fine_tune_config["max_seq_length"]
            if fine_tune_config.get("training_args") is not None:
                training_defaults.update(fine_tune_config["training_args"])
            if fine_tune_config.get("lora_config") is not None:
                lora_config_default.update(
                    fine_tune_config["lora_config"],
                )

        if fine_tune_config.get("continue_lora_finetuning") is True:
            if not isinstance(model, PeftModel):
                self.lora_config = PeftConfig.from_pretrained(
                    self.local_model_path,
                )
                model = PeftModel.from_pretrained(model, self.local_model_path)
                # unfreeze lora parameters. Assuming
                # 'lora' is in the lora layer name.
                for name, param in model.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
        else:
            if lora_config_default:
                self.lora_config = LoraConfig(**lora_config_default)
                model = get_peft_model(model, self.lora_config)

        if output_dir is not None:
            training_defaults["output_dir"] = output_dir

        trainer_args = transformers.TrainingArguments(**training_defaults)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=formatted_dataset,
            eval_dataset=formatted_dataset,
            **(
                {"peft_config": self.lora_config}
                if self.lora_config is not None
                else {}
            ),
            args=trainer_args,
            max_seq_length=max_seq_length_default,
        )

        logger.info(
            "Starting fine-tuning of the model '{model_name}' at {timestamp}",
            model_name=model.config.name_or_path,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        trainer.train()

        now = datetime.now()
        time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        # save model
        model_name = (
            f"sft_{model.config.name_or_path.split('/')[-1]}_{time_string}"
        )
        if output_dir is not None:
            model_path = os.path.join(output_dir, model_name)
        else:
            model_path = os.path.join(os.path.dirname(__file__), model_name)
        trainer.save_model(model_path)

        # save token embeddings because it is resized
        # due to the addition of new special tokens
        embed_tokens_weights = (
            model.get_input_embeddings().weight.data.clone().detach()
        )
        torch.save(
            embed_tokens_weights,
            model_path + "/embed_tokens_weights.pt",
        )

        with open(
            os.path.join(model_path, "log_history.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(trainer.state.log_history, f)

        return model
