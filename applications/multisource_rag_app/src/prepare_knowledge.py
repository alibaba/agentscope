# -*- coding: utf-8 -*-
"""
Preprocess data and insert them in ElasticSearch
"""
# pylint: disable=E0611
import json
import os
import argparse
from argparse import ArgumentParser
from loguru import logger
from copilot_app import init_model

from elasticsearch import Elasticsearch
from utils.prepare_data_utils import prepare_docstring_txt
from app_knowledge.local_knowledge import ESKnowledge

from agentscope.rag import KnowledgeBank


base_dir = os.path.dirname(__file__)


def delete_existing_es_idx(knowledge_configs: list) -> None:
    """
    Delete existing indexes in Elasticsearch to avoid duplicate records
    Args:
        knowledge_configs (list): list of knowledge configs
    """
    es = Elasticsearch("http://localhost:9200")
    for config in knowledge_configs:
        index_name = config["es_config"]["vector_store_args"]["index_name"]
        response = es.indices.delete(index=index_name, ignore=[400, 404])
        cluster_idx = index_name + "_cluster_means"
        response = es.indices.delete(index=cluster_idx, ignore=[400, 404])

        # Check response
        if "acknowledged" in response and response["acknowledged"]:
            print(f"Index {index_name} deleted successfully.")
        else:
            print(f"Failed to delete index {index_name}.")


# ES configs
knowledge_config_mapping = {
    "as": "configs/as_config/as_knowledge_configs/"
    "as_es_knowledge_update_config.json",
}


# Custom validation function
def validate_string(value: str) -> str:
    """validate if the string is a key in knowledge_config_mapping"""
    predefine_set = set(knowledge_config_mapping.keys())
    if value not in predefine_set:
        raise argparse.ArgumentTypeError(
            f"Invalid choice: {value}. "
            f"Valid choices are: {', '.join(predefine_set)}",
        )
    return value


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--repo_list",
        type=validate_string,
        nargs="+",  # Accept one or more arguments
        help=f"must be in {set(knowledge_config_mapping.keys())}",
        default=[],
    )
    args = parser.parse_args()

    print("Will update:", args.repo_list)

    knowledge_config_paths = [
        knowledge_config_mapping[r] for r in args.repo_list
    ]
    print("Using configs:", knowledge_config_paths)

    init_model(
        model_config_path=os.path.join(base_dir, "configs/model_config.json"),
    )

    if "as" in args.repo_list:
        # change the following path to the agentscope tutorial if needed
        prepare_docstring_txt(
            repo_path=os.getenv("REPO_PATH"),
            text_dir=os.getenv("TEXT_DIR"),
        )

    for knowledge_config_path in knowledge_config_paths:
        with open(knowledge_config_path, "r", encoding="utf-8") as f:
            cur_knowledge_configs = json.load(f)

        logger.info(f"\n{cur_knowledge_configs}")

        delete_existing_es_idx(cur_knowledge_configs)

    knowledge_bank = KnowledgeBank(
        configs=cur_knowledge_configs,
        new_knowledge_types=[ESKnowledge],
    )
