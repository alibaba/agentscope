# -*- coding: utf-8 -*-
"""
Knowledge bank for making Knowledge objects easier to use
"""
import copy
import json
from typing import Optional, Union
from loguru import logger
from agentscope.agents import AgentBase
from ..manager import ModelManager
from .knowledge import Knowledge

from .llama_index_knowledge import LlamaIndexKnowledge
from .langchain_knowledge import LangChainKnowledge

DEFAULT_INDEX_CONFIG = {
    "knowledge_id": "",
    "data_processing": [],
}
DEFAULT_LOADER_CONFIG = {
    "load_data": {
        "loader": {
            "create_object": True,
            "module": "llama_index.core",
            "class": "SimpleDirectoryReader",
            "init_args": {},
        },
    },
}
LANGCHAIN_LOADER_CONFIG = {
    "load_data": {
        "loader": {
            "create_object": True,
            "module": "langchain_community.document_loaders",
            "class": "DirectoryLoader",
            "init_args": {},
        },
    },
}
DEFAULT_INIT_CONFIG = {
    "input_dir": "",
    "recursive": True,
    "required_exts": [],
}
LANGCHAIN_INIT_CONFIG = {
    "path": "",
    "recursive": True,
    "glob": [],
}


class KnowledgeBank:
    """
    KnowledgeBank enables
    1) provide an easy and fast way to initialize the Knowledge object;
    2) make Knowledge object reusable and sharable for multiple agents.
    """

    def __init__(
        self,
        configs: Union[dict, str],
    ) -> None:
        """initialize the knowledge bank"""

        if isinstance(configs, str):
            logger.info(f"Loading configs from {configs}")
            with open(configs, "r", encoding="utf-8") as fp:
                self.configs = json.loads(fp.read())
        else:
            self.configs = configs
        self.stored_knowledge: dict[str, Knowledge] = {}
        self._init_knowledge()

    def _init_knowledge(self) -> None:
        """initialize the knowledge bank"""
        for config in self.configs:
            print("bank", config)
            self.add_data_as_knowledge(
                knowledge_id=config["knowledge_id"],
                emb_model_name=config["emb_model_config_name"],
                knowledge_config=config,
            )
        logger.info("knowledge bank initialization completed.\n ")

    def add_data_as_knowledge(
        self,
        knowledge_id: str,
        emb_model_name: str,
        data_dirs_and_types: dict[str, list[str]] = None,
        model_name: Optional[str] = None,
        backend_engine: Optional[str] = None,
        knowledge_config: Optional[dict] = None,
    ) -> None:
        """
        Transform data in a directory to be ready to work with RAG.
        Args:
            knowledge_id (str):
                user-defined unique id for the knowledge
            emb_model_name (str):
                name of the embedding model
            model_name (Optional[str]):
                name of the LLM for potential post-processing or query rewrite
            data_dirs_and_types (dict[str, list[str]]):
                dictionary of data paths (keys) to the data types
                (file extensions) for knowledgebase
                (e.g., [".md", ".py", ".html"])
                (if use langchain backend: ["*.md", "**/*.py", "*.html"])
            backend_engine (Optional[str]):
                name of the backend engine
                (only used for llama_index or langchain)
            knowledge_config (optional[dict]):
                complete indexing configuration, used for more advanced
                applications. Users can customize
                - loader,
                - transformations,
                - ...
                Examples can refer to../examples/conversation_with_RAG_agents/

            a simple example of importing data to Knowledge object:
            ''
                knowledge_bank.add_data_as_knowledge(
                    knowledge_id="agentscope_tutorial_rag",
                    emb_model_name="qwen_emb_config",
                    data_dirs_and_types={
                        "../../docs/sphinx_doc/en/source/tutorial": [".md"],
                    },
                    persist_dir="./rag_storage/tutorial_assist",
                )
            ''
        """

        if knowledge_id in self.stored_knowledge:
            raise ValueError(f"knowledge_id {knowledge_id} already exists.")

        assert data_dirs_and_types is not None or knowledge_config is not None

        if knowledge_config is None:
            if backend_engine is None or backend_engine == "llama_index":
                backend_engine = "llama_index"
                knowledge_config = copy.deepcopy(DEFAULT_INDEX_CONFIG)
                for data_dir, types in data_dirs_and_types.items():
                    loader_config = copy.deepcopy(DEFAULT_LOADER_CONFIG)
                    loader_init = copy.deepcopy(DEFAULT_INIT_CONFIG)
                    loader_init["input_dir"] = data_dir
                    loader_init["required_exts"] = types
                    loader_config["load_data"]["loader"][
                        "init_args"
                    ] = loader_init
                    knowledge_config["data_processing"].append(loader_config)
            elif backend_engine == "langchain":
                knowledge_config = copy.deepcopy(DEFAULT_INDEX_CONFIG)
                for data_dir, types in data_dirs_and_types.items():
                    loader_config = copy.deepcopy(LANGCHAIN_LOADER_CONFIG)
                    loader_init = copy.deepcopy(LANGCHAIN_INIT_CONFIG)
                    loader_init["path"] = data_dir
                    loader_init["glob"] = types
                    loader_config["load_data"]["loader"][
                        "init_args"
                    ] = loader_init
                    knowledge_config["data_processing"].append(loader_config)
            else:
                raise ValueError(
                    f"Backend engine {backend_engine} not supported.",
                )
        else:
            backend_engine = knowledge_config.pop("backend_engine", None)

        if backend_engine is None:
            # get the backend engine
            backend_engine = self._get_backend_engine(knowledge_config)

        if "llama_index" in backend_engine:
            CustomKnowledge = LlamaIndexKnowledge
            logger.info("Using llama_index backend engine")
        elif "langchain" in backend_engine:
            CustomKnowledge = LangChainKnowledge
            logger.info("Using langchain backend engine")
        else:
            raise ValueError(f"Backend engine {backend_engine} not supported.")

        model_manager = ModelManager.get_instance()

        self.stored_knowledge[knowledge_id] = CustomKnowledge(
            knowledge_id=knowledge_id,
            emb_model=model_manager.get_model_by_config_name(emb_model_name),
            knowledge_config=knowledge_config,
            model=(
                model_manager.get_model_by_config_name(model_name)
                if model_name
                else None
            ),
        )
        logger.info(f"data loaded for knowledge_id = {knowledge_id}.")

    def get_knowledge(
        self,
        knowledge_id: str,
        duplicate: bool = False,
    ) -> Knowledge:
        """
        Get a Knowledge object from the knowledge bank.
        Args:
            knowledge_id (str):
                unique id for the Knowledge object
            duplicate (bool):
                whether return a copy of the Knowledge object.
        Returns:
            Knowledge:
                the Knowledge object defined with Llama-index
        """
        if knowledge_id not in self.stored_knowledge:
            raise ValueError(
                f"{knowledge_id} does not exist in the knowledge bank.",
            )
        knowledge = self.stored_knowledge[knowledge_id]
        if duplicate:
            knowledge = copy.deepcopy(knowledge)
        logger.info(f"knowledge bank loaded: {knowledge_id}.")
        return knowledge

    def equip(
        self,
        agent: AgentBase,
        knowledge_id_list: list[str] = None,
        duplicate: bool = False,
    ) -> None:
        """
        Equip the agent with the knowledge by knowledge ids.

        Args:
            agent (AgentBase):
                the agent to be equipped with knowledge
            knowledge_id_list:
                the list of knowledge ids to be equipped with the agent
            duplicate (bool): whether to deepcopy the knowledge object
        TODO: to accommodate with distributed setting
        """
        logger.info(f"Equipping {agent.name} knowledge {knowledge_id_list}")
        knowledge_id_list = knowledge_id_list or []

        if not hasattr(agent, "knowledge_list"):
            agent.knowledge_list = []
        for kid in knowledge_id_list:
            knowledge = self.get_knowledge(
                knowledge_id=kid,
                duplicate=duplicate,
            )
            agent.knowledge_list.append(knowledge)

    def _get_backend_engine(self, config: dict) -> str:
        """Determines the backend engine based on the configuration.

        Iterates through the `data_processing` section of the configuration.
        It checks each process to find a dict that contains a 'module' key.
        If found, it returns the corresponding backend engine
        ('langchain' or 'llama_index').

        Args:
            config (dict):
                The configuration dictionary containing
                data processing information.

        Returns:
            str: The name of the backend engine.
        """
        data_processing = config.get("data_processing", [])

        for process in data_processing:
            if isinstance(process, dict):
                for value in process.values():
                    if isinstance(value, dict):
                        if "module" in value:
                            module_value = value["module"]
                        else:
                            module_value = self._recursive_find_module(value)

                        if module_value:
                            return module_value.split(".")[0]

        raise ValueError(
            "No rag backend engine module found, "
            "please check your knowledge config",
        )

    def _recursive_find_module(self, d: dict) -> Optional[str]:
        """Recursively searches for a 'module' key in a nested dict.

        This method traverses a nested dict and returns
        the value associated with the first 'module' key it finds.

        Args:
            d (dict): The dictionary to search.

        Returns:
            str: The value of the 'module' key, or None if not found.
        """
        for v in d.values():
            if isinstance(v, dict):
                if "module" in v:
                    return v["module"]
                module_value = self._recursive_find_module(v)
                if module_value:
                    return module_value
        return None
