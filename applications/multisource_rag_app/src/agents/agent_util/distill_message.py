# -*- coding: utf-8 -*-
"""
This module includes function(s) for shortening messages.
"""
import re
from agentscope.message import Msg


def rule_based_shorten_msg(
    origin_message: Msg,
    len_per_chunk: int = 50,
) -> Msg:
    """
    Shorten a message to at most len_per_chunk length.
    Args:
        origin_message (`Msg`):
            Original message.
        len_per_chunk (`int`):
            Length of shortened message.
    Returns:
        `Msg`: a message with processed information
    """
    # add "\n" to ensure result content not empty when there are no newlines
    content = origin_message.content + "\n"
    list_content = re.findall("(.*)\n", content)
    truncated_content = []
    for s in list_content:
        if len(s) > len_per_chunk:
            truncated_content.append(s[:len_per_chunk] + "...")
        else:
            truncated_content.append(s)
    return Msg(
        name=origin_message.name,
        role=origin_message.role,
        content="\n".join(truncated_content),
    )
