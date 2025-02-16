# -*- coding: utf-8 -*-
"""Logging utilities."""
import os
import sys
from typing import Optional, Literal, Any

from loguru import logger


from .message import Msg
from .serialize import serialize
from .studio._client import _studio_client
from .utils.common import _guess_type_by_extension
from .web.gradio.utils import (
    generate_image_from_name,
    send_msg,
    get_reset_msg,
    thread_local_data,
)

LOG_LEVEL = Literal[
    "TRACE",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
]

LEVEL_SAVE_LOG = "SAVE_LOG"
LEVEL_SAVE_MSG = "SAVE_MSG"

_DEFAULT_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{"
    "level: <8}</level> | <cyan>{name}</cyan>:<cyan>{"
    "function}</cyan>:<cyan>{line}</cyan> - <level>{"
    "message}</level>"
)

_PREFIX_DICT = {}


def log_stream_msg(msg: Msg, last: bool = True) -> None:
    """Print the message in different streams, including terminal, studio, and
    gradio if it is active.

    Args:
        msg (`Msg`):
            The message object to be printed.
        last (`bool`, defaults to `True`):
            True if this is the last message in the stream or a single message.
            Otherwise, False.
    """
    global _PREFIX_DICT

    # Print msg to terminal
    formatted_str = msg.formatted_str(colored=True)
    
    # Debug prefix tracking
    # prev_len = _PREFIX_DICT.get(msg.id, 0)
    # print(f"\n[DEBUG] Previous length: {prev_len}")
    # print(f"[DEBUG] Current length: {len(formatted_str)}")
    # print(f"[DEBUG] Content being sliced: '{formatted_str}'")
    # print(f"[DEBUG] Slice result: '{formatted_str[prev_len:]}'")
    
    print_str = formatted_str[_PREFIX_DICT.get(msg.id, 0) :]

    if last:
        # Remove the prefix from the dictionary
        del _PREFIX_DICT[msg.id]

        print(print_str)
    else:
        # Update the prefix in the dictionary
        _PREFIX_DICT[msg.id] = len(formatted_str)

        print(print_str, end="")

    # Push msg to studio if it is active
    if _studio_client.active:
        _studio_client.push_message(msg)

    # Print to gradio if it is active
    if last and hasattr(thread_local_data, "uid"):
        log_gradio(msg, thread_local_data.uid)

    if last:
        # Save msg into chat file
        _save_msg(msg)


def _save_msg(msg: Msg) -> None:
    """Save the message into `logging.chat` and `logging.log` files.

    Args:
        msg (`Msg`):
            The message object to be saved.
    """
    # TODO: Unified into a manager rather than an indicated attribute here
    if hasattr(logger, "chat"):
        # Not initialize yet
        logger.log(
            LEVEL_SAVE_LOG,
            msg.formatted_str(colored=False),
        )

        logger.log(
            LEVEL_SAVE_MSG,
            serialize(msg),
        )


def log_msg(msg: Msg, disable_gradio: bool = False) -> None:
    """Print the message and save it into files. Note the message should be a
    Msg object."""

    if not isinstance(msg, Msg):
        raise TypeError(f"Get type {type(msg)}, expect Msg object.")

    print(msg.formatted_str(colored=True))

    # Push msg to studio if it is active
    if _studio_client.active:
        _studio_client.push_message(msg)

    # Print to gradio if it is active
    if hasattr(thread_local_data, "uid") and not disable_gradio:
        log_gradio(msg, thread_local_data.uid)

    # Save msg into chat file
    _save_msg(msg)


def log_gradio(msg: Msg, uid: str, **kwargs: Any) -> None:
    """Send chat message to studio.

    Args:
        msg (`Msg`):
            The message to be logged.
        uid (`str`):
            The local value 'uid' of the thread.
    """
    if uid:
        get_reset_msg(uid=uid)
        avatar = kwargs.get("avatar", None) or generate_image_from_name(
            msg.name,
        )

        content = msg.content
        flushing = True
        if msg.url is not None:
            flushing = False
            if isinstance(msg.url, str):
                urls = [msg.url]
            else:
                urls = msg.url

            for url in urls:
                typ = _guess_type_by_extension(url)
                if typ == "image":
                    content += f"\n<img src='{url}'/>"
                elif typ == "audio":
                    content += f"\n<audio src='{url}' controls/></audio>"
                elif typ == "video":
                    content += f"\n<video src='{url}' controls/></video>"
                else:
                    content += f"\n<a href='{url}'>{url}</a>"

        send_msg(
            content,
            role=msg.name,
            uid=uid,
            flushing=flushing,
            avatar=avatar,
        )


def _level_format(record: dict) -> str:
    """Format the log record."""
    # Display the chat message
    if record["level"].name == LEVEL_SAVE_LOG:
        return "{message}\n"
    else:
        return _DEFAULT_LOG_FORMAT + "\n"


def setup_logger(
    path_log: Optional[str] = None,
    level: LOG_LEVEL = "INFO",
) -> None:
    r"""Setup `loguru.logger` and redirect stderr to logging.

    Args:
        path_log (`str`, defaults to `""`):
            The directory of log files.
        level (`str`, defaults to `"INFO"`):
            The logging level, which is one of the following: `"TRACE"`,
            `"DEBUG"`, `"INFO"`, `"SUCCESS"`, `"WARNING"`, `"ERROR"`,
            `"CRITICAL"`.
    """
    # set logging level
    logger.remove()

    # avoid reinit in subprocess
    if not hasattr(logger, "chat"):
        # add chat function for logger
        logger.level(LEVEL_SAVE_LOG, no=51)

        # save chat message into file
        logger.level(LEVEL_SAVE_MSG, no=53)
        logger.chat = log_msg

        # standard output for all logging except chat
        logger.add(
            sys.stdout,
            filter=lambda record: record["level"].name
            not in [LEVEL_SAVE_LOG, LEVEL_SAVE_MSG],
            format=_DEFAULT_LOG_FORMAT,
            enqueue=True,
            level=level,
        )

    if path_log is not None:
        os.makedirs(path_log, exist_ok=True)
        path_log_file = os.path.join(path_log, "logging.log")
        path_chat_file = os.path.join(
            path_log,
            "logging.chat",
        )

        # save all logging except LEVEL_SAVE_MSG into logging.log
        logger.add(
            path_log_file,
            filter=lambda record: record["level"].name != LEVEL_SAVE_MSG,
            format=_level_format,
            enqueue=True,
            level=level,
        )

        # save chat message into logging.chat
        logger.add(
            path_chat_file,
            format="{message}",
            enqueue=True,
            level=LEVEL_SAVE_MSG,  # The highest level to filter out all
            # other logs
        )
