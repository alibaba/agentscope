# -*- coding: utf-8 -*-
"""
This module provides the Finetune_DialogAgent class,
which extends DialogAgent to enhance fine-tuning
capabilities with custom hyperparameters.
"""
from typing import Any, Optional, Dict

from loguru import logger

from agentscope.agents import DialogAgent


class Finetune_DialogAgent(DialogAgent):
    """
    A dialog agent capable of fine-tuning its
    underlying model based on provided data.

    Inherits from DialogAgent and adds functionality for
    fine-tuning with custom hyperparameters.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        use_memory: bool = True,
        memory_config: Optional[dict] = None,
    ):
        """
        Initializes a new Finetune_DialogAgent with specified configuration.

        Arguments:
            name (str): Name of the agent.
            sys_prompt (str): System prompt or description of the agent's role.
            model_config_name (str): The configuration name for
                                     the underlying model.
            use_memory (bool, optional): Indicates whether to utilize
                                         memory features. Defaults to True.
            memory_config (dict, optional): Configuration for memory
                                            functionalities if
                                            `use_memory` is True.

        Note:
            Refer to `class DialogAgent(AgentBase)` for more information.
        """

        super().__init__(
            name,
            sys_prompt,
            model_config_name,
            use_memory,
            memory_config,
        )

    def load_model(
        self,
        model_id: Optional[str] = None,
        local_model_path: Optional[str] = None,
    ) -> None:
        """
        Load a new model into the agent.

        Arguments:
            model_id (str): The Hugging Face model ID or a custom identifier.
                            Needed if loading model from Hugging Face.
            local_model_path (str, optional): Path to a locally saved model.

        Raises:
            Exception: If the model loading process fails or if the
                       model wrapper does not support dynamic loading.
        """

        if hasattr(self.model, "load_model"):
            self.model.load_model(model_id, local_model_path)
        else:
            logger.error(
                "The model wrapper does not support dynamic model loading.",
            )

    def load_tokenizer(
        self,
        model_id: Optional[str] = None,
        local_tokenizer_path: Optional[str] = None,
    ) -> None:
        """
        Load a new tokenizer for the agent.

        Arguments:
            model_id (str): The Hugging Face model ID or a custom identifier.
                            Needed if loading tokenizer from Hugging Face.
            local_tokenizer_path (str, optional): Path to a locally saved
                                                  tokenizer.

        Raises:
            Exception: If the model tokenizer process fails or if the
                       model wrapper does not support dynamic loading.
        """

        if hasattr(self.model, "load_tokenizer"):
            self.model.load_tokenizer(model_id, local_tokenizer_path)
        else:
            logger.error("The model wrapper does not support dynamic loading.")

    def fine_tune(
        self,
        data_path: Optional[str] = None,
        fine_tune_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Fine-tune the agent's underlying model.

        Arguments:
            data_path (str): The path to the training data.

        Raises:
            Exception: If fine-tuning fails or if the
                       model wrapper does not support fine-tuning.
        """

        if hasattr(self.model, "fine_tune"):
            self.model.fine_tune(data_path, fine_tune_config)
            logger.info("Fine-tuning completed successfully.")
        else:
            logger.error("The model wrapper does not support fine-tuning.")
