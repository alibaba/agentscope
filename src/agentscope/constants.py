# -*- coding: utf-8 -*-
""" Some constants used in the project"""
import os
from numbers import Number
from enum import IntEnum

from pathlib import Path

PACKAGE_NAME = "agentscope"
MSG_TOKEN = f"[{PACKAGE_NAME}_msg]"


# default values
_RUNTIME_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_RUNTIME_ID_FORMAT = "run_%Y%m%d-%H%M%S_{}"

# for file manager
_DEFAULT_SAVE_DIR = "./runs"
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_SUBDIR_CODE = "code"
_DEFAULT_SUBDIR_FILE = "file"
_DEFAULT_SUBDIR_INVOKE = "invoke"
_DEFAULT_CACHE_DIR = str(
    Path(
        os.environ.get(
            "AS_HOME_PATH",
            str(Path.home()),
        ),
    )
    / ".cache"
    / "agentscope",
)
_DEFAULT_CFG_NAME = ".config"
_DEFAULT_IMAGE_NAME = "image_{}_{}.png"
_DEFAULT_SQLITE_DB_NAME = "agentscope.db"


# for model wrapper
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_MESSAGES_KEY = "messages"
_DEFAULT_RETRY_INTERVAL = 1
_DEFAULT_API_BUDGET = None
# for execute python
_DEFAULT_PYPI_MIRROR = "http://mirrors.aliyun.com/pypi/simple/"
_DEFAULT_TRUSTED_HOST = "mirrors.aliyun.com"
# for monitor
_DEFAULT_TABLE_NAME_FOR_CHAT_AND_EMBEDDING = "chat_and_embedding_model_monitor"
_DEFAULT_TABLE_NAME_FOR_IMAGE = "image_model_monitor"
# for summarization
_DEFAULT_SUMMARIZATION_PROMPT = """
TEXT: {}
"""
_DEFAULT_SYSTEM_PROMPT = """
You are a helpful agent to summarize the text.
You need to keep all the key information of the text in the summary.
"""
_DEFAULT_TOKEN_LIMIT_PROMPT = """
Summarize the text after TEXT in less than {} tokens:
"""

# typing
Embedding = list[Number]

# rpc

# set max message size to 32 MB
_DEFAULT_RPC_OPTIONS = [
    ("grpc.max_send_message_length", 32 * 1024 * 1024),
    ("grpc.max_receive_message_length", 32 * 1024 * 1024),
    ("grpc.max_metadata_size", 64 * 1024),
]
_DEFAULT_RPC_TIMEOUT = 5
_DEFAULT_RPC_RETRY_TIMES = 10


# enums
class ResponseFormat(IntEnum):
    """Enum for model response format."""

    NONE = 0
    JSON = 1


class ShrinkPolicy(IntEnum):
    """Enum for shrink strategies when the prompt is too long."""

    TRUNCATE = 0
    SUMMARIZE = 1


# rag related
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.4

# flask server
EXPIRATION_SECONDS = 604800  # One week
TOKEN_EXP_TIME = 1440  # One day long
FILE_SIZE_LIMIT = 1024 * 1024  # 10 MB
FILE_COUNT_LIMIT = 10
