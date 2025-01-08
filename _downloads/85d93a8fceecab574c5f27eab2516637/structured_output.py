# -*- coding: utf-8 -*-
"""
.. _structured-output:

Structured Output
==========================

In this tutorial, we will be building a simple agent that outputs structured
data in JSON dictionary format using the `agentscope.parsers` module.
"""
from agentscope.models import ModelResponse

# %%
# Defining the Parser
# -------------------

from agentscope.parsers import MarkdownJsonDictParser


parser = MarkdownJsonDictParser(
    content_hint='{"thought": "What you thought", "speak": "What you speak to the user"}',
    required_keys=["thought", "speak"],
)


# %%
# The parser will generate a format instruction according to your input. You
# can use the `format_instruction` property to in your prompt to guide LLM to
# generate the desired output.

print(parser.format_instruction)

# %%
# Parsing the Output
# -------------------
# When receiving output from LLM, use `parse` method to extract the
# structured data.
# It takes an object of `agentscope.models.ModelResponse` as input, parses
# the value of the `text` field, and returns a parsed dictionary in the
# `parsed` field.

dummy_response = ModelResponse(
    text="""```json
{
    "thought": "I should greet the user",
    "speak": "Hi! How can I help you?"
}
```""",
)

print(f"parsed field before parsing: {dummy_response.parsed}")

parsed_response = parser.parse(dummy_response)

print(f"parsed field after parsing: {parsed_response.parsed}")
print(type(parsed_response.parsed))

# %%
# Error Handling
# -------------------
# If the LLM output does not match the expected format, the parser will raise
# an error with a detailed message.
# So developers can present the error message to LLM to guide it to correct
# the output.

error_response = ModelResponse(
    text="""```json
{
    "thought": "I should greet the user"
}
```""",
)

try:
    parsed_response = parser.parse(error_response)
except Exception as e:
    print(e)

# %%
# Advanced Usage
# -------------------
# More Complex Content
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Asking LLM to directly generate a JSON dictionary can be challenging,
# especially when the JSON content is complex (e.g. code snippets, nested
# structures).
# In this case, you can use more advanced parsers to guide LLM to generate
# the desired output.
# Here is an example of a more complex parser that handle code snippets.

from agentscope.parsers import RegexTaggedContentParser

parser = RegexTaggedContentParser(
    format_instruction="""Response in the following format:
<thought>what you thought</thought>
<number>A random number here</number>
<code>your python code here</code>
""",
    try_parse_json=True,  # Try to parse the each value as a JSON object
    required_keys=[
        "thought",
        "number",
        "code",
    ],  # Required keys in the parsed dictionary
)

print(parser.format_instruction)

# %%
# The `RegexTaggedContentParser` uses regular expressions to match the tagged
# content in the text and return the parsed dictionary.
#
# .. note:: The parsed output of `RegexTaggedContentParser` is a dictionary, which means the required keys should be unique.
# You can also change the regular expression pattern by settings the `tagged_content_pattern` parameter when initializing the parser.

import json

dummy_response = ModelResponse(
    text="""<thought>Print the current date</thought>
<number>42</number>
<code>import datetime
print(datetime.datetime.now())
</code>
""",
)

parsed_response = parser.parse(dummy_response)

print("The type of parsed response: ", type(parsed_response.parsed))
print("The type of the number: ", type(parsed_response.parsed["number"]))
print(json.dumps(parsed_response.parsed, indent=4))

# %%
# Auto Post-Processing
# ^^^^^^^^^^^^^^^^^^^^
#
# Within the parsed dictionary, different keys may require different
# post-processing steps.
# For example, in a werewolf game, the LLM is playing the role of a seer, and
# the output should contain the following keys:
#
# - `thought`: The seer's thoughts
# - `speak`: The seer's speech
# - `use_ability`: A boolean value indicating whether the seer should use its ability
#
# In this case, the `thought` and `speak` contents should be stored in the
# agent's memory to ensure the consistency of the agent's behavior.
# The `speak` content should be spoken out to the user.
# The `use_ability` key should be accessed outside the agent easily to
# determine the game flow.
#
# AgentScope supports automatic post-processing of the parsed dictionary by
# providing the following parameters when initializing the parser.
#
# - `keys_to_memory`: key(s) that should be stored in the agent's memory
# - `keys_to_content`: key(s) that should be spoken out
# - `keys_to_metadata`: key(s) that should be stored in the metadata field of the agent's response message
#
# .. note:: If a string is provided, the parser will extract the value of the given key from the parsed dictionary. If a list of strings is provided, a sub-dictionary will be created with the given keys.
#
# Here is an example of using the `MarkdownJsonDictParser` to automatically
# post-process the parsed dictionary.

parser = MarkdownJsonDictParser(
    content_hint='{"thought": "what you thought", "speak": "what you speak", "use_ability": "whether to use the ability"}',
    keys_to_memory=["thought", "speak"],
    keys_to_content="speak",
    keys_to_metadata="use_ability",
)

dummy_response = ModelResponse(
    text="""```json
{
    "thought": "I should ...",
    "speak": "I will not use my ability",
    "use_ability": false
}```
""",
)

parsed_response = parser.parse(dummy_response)

print("The parsed response: ", parsed_response.parsed)
print("To memory", parser.to_memory(parsed_response.parsed))
print("To message content: ", parser.to_content(parsed_response.parsed))
print("To message metadata: ", parser.to_metadata(parsed_response.parsed))

# %%
# Here we show how to create an agent that can automatically post-process the
# parsed dictionary by the following core steps in the `reply` method.
#
# 1. Put the format instruction in prompt to guide LLM to generate the desired output
# 2. Parse the LLM response
# 3. Post-process the parsed dictionary by using the `to_memory`, `to_content`, and `to_metadata` methods
#
# .. tip:: By changing different parsers, the agent can adapt to different scenarios and generate structured output in various formats.

from agentscope.models import DashScopeChatWrapper
from agentscope.agents import AgentBase
from agentscope.message import Msg


class Agent(AgentBase):
    def __init__(self):
        self.name = "Alice"
        super().__init__(name=self.name)

        self.sys_prompt = f"You're a helpful assistant named {self.name}."

        self.model = DashScopeChatWrapper(
            config_name="_",
            model_name="qwen-max",
        )

        self.parser = MarkdownJsonDictParser(
            content_hint='{"thought": "what you thought", "speak": "what you speak", "use_ability": "whether to use the ability"}',
            keys_to_memory=["thought", "speak"],
            keys_to_content="speak",
            keys_to_metadata="use_ability",
        )

        self.memory.add(Msg("system", self.sys_prompt, "system"))

    def reply(self, msg):
        self.memory.add(msg)

        prompt = self.model.format(
            self.memory.get_memory(),
            # Instruct the model to respond in the required format
            Msg("system", self.parser.format_instruction, "system"),
        )

        response = self.model(prompt)

        parsed_response = self.parser.parse(response)

        self.memory.add(
            Msg(
                name=self.name,
                content=self.parser.to_memory(parsed_response.parsed),
                role="assistant",
            ),
        )

        return Msg(
            name=self.name,
            content=self.parser.to_content(parsed_response.parsed),
            role="assistant",
            metadata=self.parser.to_metadata(parsed_response.parsed),
        )
