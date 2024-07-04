import os
from typing import Dict, List, Optional

import jsonlines
import tiktoken
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pygptlink.gpt_logging import logger


class GPTContext:
    """ Represents a context history usable for chat completion.

    The persona file contains a text priming the agent to act a specific way.
    The context file contains a JSONL file with previous message objects. This is used to persist the context and can also be used for post processing the potentially long history for fine-tuning.
    The context_log_file is a debug log of all received and sent message objects. This is different from the context file by the fact that it contains the full completion response, not just the bits necessary for the next chat completion.
    """

    def __init__(self, model: str, max_tokens: int, max_response_tokens: int, persona_file: Optional[str] = None,
                 context_file: Optional[str] = None, completion_log_file: Optional[str] = None) -> None:

        self.model = model
        self.token_encoding = tiktoken.encoding_for_model(model)
        self.persona_file = persona_file
        self.completion_log_file = completion_log_file or None
        self.max_tokens = max_tokens
        self.max_response_tokens = max_response_tokens

        self.context = []
        self.context_file = context_file or None
        if self.context_file:
            try:
                # Generously assume that one token is always 8 characters (it's ~4 on average in English)
                # to get an upper bound on the number of characters to load into memory. This is not
                # intended to be accurate, but rather just a way to bound the amount of memory and work
                # required to load the context.
                max_chars_to_read = max_tokens*8

                with open(self.context_file, 'r', encoding='utf-8') as file:
                    start_offs = max(0, os.path.getsize(
                        self.context_file) - max_chars_to_read)
                    file.seek(start_offs)
                    if start_offs != 0:
                        file.readline()  # Discard until next endline.
                    with jsonlines.Reader(file) as reader:
                        for message in reader.iter():
                            self.context.append(message)
            except FileNotFoundError:
                logger.warning(
                    f"The context file does not exist: {self.context_file}")
            except OSError as e:
                logger.error(f"An error occurred while opening the file: {e}")

    def append_completion(self, completion: ChatCompletion, choice: int = 0, name: str = None):
        if self.completion_log_file:
            with open(self.completion_log_file, 'a', encoding='utf-8') as file:
                file.write(completion.model_dump_json(indent=2) + '\n')

        if len(completion.choices) < 1:
            logger.warning("Empty choices for completion!")
            return
        completion_message: ChatCompletionMessage = completion.choices[choice].message

        message = {"role": completion_message.role}
        if completion_message.content is not None:
            message["content"] = completion_message.content
        if name is not None:
            message["name"] = name
        if completion_message.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in completion_message.tool_calls
            ]
        self.__append_message(message)

    def append_user_prompt(self, user: str, content: str):
        message = {"role": "user", "name": user, "content": content}
        self.__append_message(message)

    def append_tool_response(self, id, content):
        message = {"role": "tool", "content": content, "tool_call_id": id}
        self.__append_message(message)

    def append_system_message(self, message):
        self.__append_message(self.__system_message(message))

    @staticmethod
    def filter_tool_references(message, tool_id) -> Optional[Dict]:
        if message["role"] == "assistant":
            for tool_call in message["tool_calls"] or []:
                if tool_call["id"] == tool_id:
                    message["tool_calls"].remove(tool_call)
                    if len(message["tool_calls"]) == 0:
                        return None
                    break
            return message
        if message["role"] == "tool" and message["tool_call_id"] == tool_id:
            return None
        return message

    def messages(self, additional_system_prompt: str = None) -> List[Dict]:
        assert additional_system_prompt == None or isinstance(
            additional_system_prompt, str)

        if self.persona_file:
            with open(self.persona_file, 'r') as file:
                persona = file.read()

        # Find split point
        left_split = []
        right_split = []
        for entry in reversed(self.context):
            if not right_split:
                right_split.append(entry)
            elif right_split[-1]["role"] == "tool":
                right_split.append(entry)
            else:
                left_split.append(entry)
        right_split.reverse()
        left_split.reverse()

        messages = left_split
        if self.persona_file:
            messages.append(self.__system_message(persona))
        if additional_system_prompt:
            messages.append(self.__system_message(additional_system_prompt))
        messages += right_split

        available_tokens = self.max_tokens - self.max_response_tokens
        while len(messages) and (messages[0]["role"] == "tool" or self.__num_tokens_from_messages(messages) > available_tokens):
            if len(self.context) > 0:
                self.context.pop(0)
            messages.pop(0)
        return messages

    def clear(self):
        self.context = []
        if self.context_file and os.path.exists(self.context_file):
            os.remove(self.context_file)

    def __append_message(self, message):
        self.context.append(message)
        if self.context_file:
            with jsonlines.open(self.context_file, 'a') as file:
                file.write(message)

    def __system_message(self, content: str):
        return {"role": "system", "content": content}

    # Taken from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def __num_tokens_from_messages(self, messages):
        """Return the number of tokens used by a list of messages."""
        if self.model in {
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0125-preview",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif self.model == "gpt-3.5-turbo-0301":
            # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_message = 4
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {self.model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

        def recursive_iterate(dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    yield from recursive_iterate(value)
                else:
                    yield key, value

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == "tool_calls":
                    for tool_call in value:
                        for key, value in recursive_iterate(tool_call):
                            num_tokens += len(self.token_encoding.encode(value))
                else:
                    if key == "name":
                        num_tokens += tokens_per_name
                    num_tokens += len(self.token_encoding.encode(value))

        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
