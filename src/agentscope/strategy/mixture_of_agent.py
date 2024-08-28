# -*- coding: utf-8 -*-
# pylint: disable=C0301
""" Utils for mixing model's answers in agentscope """

from typing import Union, List, Sequence, Tuple
import concurrent.futures
from loguru import logger

from agentscope.manager import ModelManager
from agentscope.message import Msg
from agentscope.models import ModelWrapperBase


# Referenced from the project [MoA](https://github.com/togethercomputer/MoA)
DEFAULT_AGGREGATOR_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""  # noqa


class MixtureOfAgents:
    """
    The MoA model that take multiple models and aggregate their responses,
    leverages the collective strengths of multiple LLMs to enhance performance.
    Reference from the project [MoA](https://github.com/togethercomputer/MoA).
    """

    def __init__(
        self,
        main_model: Union[str, ModelWrapperBase],
        reference_models: List[Union[str, ModelWrapperBase]],
        rounds: int = 1,
        aggregator_prompt: str = DEFAULT_AGGREGATOR_PROMPT,
        show_internal: bool = False,
    ) -> None:
        """
        Args:
            main_model (`Union[str, ModelWrapperBase]`):
                The main_model will make the final aggregation in the last
                round, summarizing all the previous responses from models.
                Can take both config name of model or model instance as input.
            reference_models (`List[Union[str, ModelWrapperBase]]`):
                The reference_models used for generating different responses
                in each round.
                Can take both config name of model or model instance as input.
                We encourage using different models to get better diversity.
                Empirically, responses generated by heterogeneous models
                contribute more than those produced by the same model.
            rounds (`int`):
                The number of processing rounds to refine the responses.
                Can range from 0 to inf.
            aggregator_prompt (`str`):
                The prompt used for aggregating responses.
                Using the prompt from paper MoA by default.
            show_internal (`bool`):
                Whether to show the internal process of MoA.
        """
        model_manager = ModelManager.get_instance()

        # init main_model
        if isinstance(main_model, str):
            self.main_model = model_manager.get_model_by_config_name(
                main_model,
            )
        elif isinstance(main_model, ModelWrapperBase):
            self.main_model = main_model
        else:
            raise ValueError(
                "main_model must be a string or a ModelWrapperBase instance",
            )

        # init reference_models
        self.reference_models: List[ModelWrapperBase] = []
        for ref_model in reference_models:
            if isinstance(ref_model, str):
                self.reference_models.append(
                    model_manager.get_model_by_config_name(ref_model),
                )
            elif isinstance(ref_model, ModelWrapperBase):
                self.reference_models.append(ref_model)
            else:
                raise ValueError(
                    "reference_models must be a list of strings "
                    "or ModelWrapperBase instances",
                )
        self.references: List[str] = [
            "" for _ in range(len(self.reference_models))
        ]
        self.rounds = rounds
        self.aggregator_prompt = aggregator_prompt
        self.show_internal = show_internal

    def _get_res_with_aggregate_model(
        self,
        aggre_model: ModelWrapperBase,
    ) -> str:
        messages = []
        messages.append(
            Msg(role="system", content=self.aggregator_prompt, name="system"),
        )
        for i, ref in enumerate(self.references, start=0):
            messages.append(
                Msg(
                    role="user",
                    content=ref,
                    name=f"Model_{i}",
                ),
            )
        aggre_format_msg = aggre_model.format(messages)
        aggre_res = aggre_model(aggre_format_msg)
        return aggre_res.text

    def __call__(
        self,
        *args: Union[Msg, Sequence[Msg]],
    ) -> str:
        """
        Get model response from messages.
        Is equivalent to calling a model with:
            ```
            format_msg = model.format(messages)
            return model(format_msg)
            ```

        Args:
            *args (`Union[Msg, Sequence[Msg]]`):
                The messages to be sent to the model.
        """

        def _process_reference(
            i: int,
            ref_model: ModelWrapperBase,
            *args: Union[Msg, Sequence[Msg]],
        ) -> Tuple[int, str]:
            format_msg = ref_model.format(*args)
            ref_model_res = ref_model(format_msg)
            return i, ref_model_res.text

        def _process_new_refs(
            i: int,
            ref_model: ModelWrapperBase,
        ) -> Tuple[int, str]:
            return i, self._get_res_with_aggregate_model(ref_model)

        # get all the references
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_process_reference, i, ref_model, *args)
                for i, ref_model in enumerate(self.reference_models, start=0)
            ]
            for future in concurrent.futures.as_completed(futures):
                i, result = future.result()
                self.references[i] = result
                if self.show_internal:
                    logger.info(f"Round {0}, Model_{i}: {result}")

        for r in range(self.rounds):
            if self.show_internal:
                logger.info("=" * 20)
            new_refs = ["" for _ in range(len(self.reference_models))]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(_process_new_refs, i, ref_model)
                    for i, ref_model in enumerate(
                        self.reference_models,
                        start=0,
                    )
                ]
                for future in concurrent.futures.as_completed(futures):
                    i, result = future.result()
                    new_refs[i] = result
                    if self.show_internal:
                        print(f"Round {r+1}, Model_{i}: {result}")
            self.references = new_refs

        final_res = self._get_res_with_aggregate_model(self.main_model)
        return final_res
