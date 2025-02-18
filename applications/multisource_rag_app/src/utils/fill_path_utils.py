# -*- coding: utf-8 -*-
"""
Path filling helper functions
"""
import os


def filling_paths_in_knowledge_config(config: dict, base_dir: str) -> None:
    """
    path filling helper function for knowledge configs in RAG application,
    convert relative path to full path
    Args:
        config (dict): configuration dictionary
        base_dir (str): path to base directory of RAG application
    """
    data_process_configs = config.get("data_processing", [])
    for cfg in data_process_configs:
        if "input_dir" in cfg.get("load_data", {}).get("loader", {}).get(
            "init_args",
            {},
        ):
            if not cfg["load_data"]["loader"]["init_args"][
                "input_dir"
            ].startswith(base_dir):
                path = cfg["load_data"]["loader"]["init_args"]["input_dir"]
                if path.startswith("~"):
                    path = os.path.expanduser(path)
                cfg["load_data"]["loader"]["init_args"][
                    "input_dir"
                ] = os.path.join(
                    base_dir,
                    path,
                )


def fill_paths_in_agent_configs(config: dict, base_dir: str) -> None:
    """
    path filling helper function for agent configs in RAG application,
    convert relative path to full path
    Args:
        config (dict): configuration dictionary
        base_dir (str): path to base directory of RAG application
    """
    if "local_pattern" in config.get("args", {}).get("web_path_mapping", {}):
        if (
            not config["args"]["web_path_mapping"]["local_pattern"].startswith(
                base_dir,
            )
            and os.getenv("RESET_LOCAL_PATTERN", "False") == "True"
        ):
            config["args"]["web_path_mapping"]["local_pattern"] = os.path.join(
                base_dir,
                config["args"]["web_path_mapping"]["local_pattern"],
            )
