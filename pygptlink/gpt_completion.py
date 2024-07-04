import asyncio
import time
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pygptlink.gpt_context import GPTContext
from pygptlink.gpt_tool_definition import GPTToolDefinition, NoResponseDesired
from pygptlink.gpt_logging import logger
from pygptlink.sentenceextractor import SentenceExtractor


class GPTNoResponseNeeded(Exception):
    def __init__(self, message="No Response Needed"):
        self.message = message
        super().__init__(self.message)


class GPTCompletion:
    def __init__(self, api_key: str, context: GPTContext, tools: List[GPTToolDefinition] = []):
        self.lock = asyncio.Lock()
        self.oai = AsyncOpenAI(api_key=api_key)
        self.sentence_extractor = SentenceExtractor()
        self.context = context
        self.tools = {tool.name: tool for tool in tools}
        self.completion_settings = {
            'model': context.model,
            'max_tokens': context.max_response_tokens,
            'stream': True,
        }

    async def system(self, query: str, callback: Callable[[str, bool], None] = None, extra_system_prompt: Optional[str] = None,
                     force_tool: Optional[str] = None, allowed_tools: Optional[List[str]] = None) -> str:
        async with self.lock:
            self.context.append_system_message(query)
            return await self.__complete(callback=callback, extra_system_prompt=extra_system_prompt, force_tool=force_tool, allowed_tools=allowed_tools)

    async def prompt(self, speaker: str, query: str, callback: Callable[[str, bool], None] = None, extra_system_prompt: Optional[str] = None,
                     force_tool: Optional[str] = None, allowed_tools: Optional[List[str]] = None) -> str:
        async with self.lock:
            self.context.append_user_prompt(speaker, query)
            return await self.__complete(callback=callback, extra_system_prompt=extra_system_prompt, force_tool=force_tool, allowed_tools=allowed_tools)

    async def __complete(self,
                         callback: Callable[[str, bool], None],
                         extra_system_prompt: Optional[str] = None,
                         force_tool: Optional[str] = None,
                         allowed_tools: Optional[List[str]] = None) -> str:
        """Generates a response to the current context.

        Args:
            extra_system_prompt (Optional[str]): An extra system prompt to be injected into the context, can be None.
            callback (Callable[[str], None]): A callback to call with the response, line by line.
            force_tool (str, optional): The name of a tool that must be called by the model. Defaults to None.
            allowed_tools (List[str], optional): A list of tools that the model may call. None means any tool it
                                                 knows of, [] means no tools may be called, or a list of named
                                                 tools. Defaults to None.

        Raises:
            ValueError: When inputs are invalid.

        Returns:
            Nothing
        """
        try:
            # Prepare arguments for completion
            messages = self.context.messages(extra_system_prompt)
            logger.debug(f"Prompting with: {messages}")
            if force_tool:
                if not force_tool in self.tools:
                    logger.error(f"Unknown tool in force_tool! {force_tool}")
                    raise ValueError(f"Non existent tool {force_tool} forced.")
                tool_choice = {
                    "type": "function",
                    "function": {"name": force_tool}
                }
                tool_defs = [self.tools[force_tool].describe()]
            else:
                if allowed_tools == None:
                    tool_defs = [
                        tool.describe() for tool in self.tools.values()] if self.tools else None
                    tool_choice = "auto" if self.tools else None
                elif allowed_tools == []:
                    tool_defs = None
                    tool_choice = None
                else:
                    if not all(tool_name in self.tools for tool_name in allowed_tools):
                        logger.error(
                            f"Unknown tool in allowed tools! {allowed_tools}")
                        raise ValueError("Invalid allowed tools list")
                    tool_defs = [self.tools[tool_name].describe()
                                 for tool_name in allowed_tools]
                    tool_choice = "auto"

            # Stream the completion
            stream = await self.oai.chat.completions.create(messages=messages, **self.completion_settings, tools=tool_defs, tool_choice=tool_choice)
            partial_sentence = ""
            full_response: Dict[str, str] = {}
            chunk: ChatCompletionChunk
            async for chunk in stream:
                GPTCompletion.__merge_dicts(full_response, chunk.model_dump())
                partial_sentence += chunk.choices[0].delta.content or ""
                lines, partial_sentence = self.sentence_extractor.extract_partial(
                    partial_sentence)
                for line in lines:
                    if callback:
                        callback(line, False)
            partial_sentence = partial_sentence.strip()
            if callback:
                callback(partial_sentence, True)

            # Look for any function calls in the finished completion.
            chat_completion = GPTCompletion.__to_chat_completion(full_response)
            logger.debug(f"Received object: {chat_completion}")
            self.context.append_completion(chat_completion)
            should_respond_to_tool = False
            for choice in chat_completion.choices:
                for tool_call in choice.message.tool_calls or []:
                    tool = self.tools.get(tool_call.function.name, None)
                    if tool is None:
                        logger.warn(
                            f"Invalid tool invocation, tool: {tool_call.function.name} doesn't exist.")
                        response = f"Error: No such tool: {tool_call.function.name}."
                        self.context.append_tool_response(
                            tool_call.id, response)
                        # Let the LLM know so it can try to fix
                        should_respond_to_tool = True
                    else:
                        logger.info(f"Tool invocation: {tool_call.function}")
                        response = await tool.invoke(tool_call)
                        if isinstance(response, NoResponseDesired):
                            response = ""
                        else:
                            should_respond_to_tool = True
                        self.context.append_tool_response(
                            tool_call.id, response)
                logger.info(
                    f" -- LLM Response Ended ({choice.finish_reason}) -- ")

            response = None
            if chat_completion.choices[0].message:
                response = chat_completion.choices[0].message.content

            if should_respond_to_tool:
                sub_response = await self.__complete(extra_system_prompt=extra_system_prompt, callback=callback, allowed_tools=allowed_tools, force_tool=force_tool)

                if not response:
                    return sub_response
                elif sub_response:
                    return response + " " + sub_response
                else:
                    return response
            else:
                return response

        except openai.APIConnectionError as e:
            logger.error("The server could not be reached")
            logger.error(e.__cause__)
        except openai.RateLimitError as e:
            logger.warn(
                "A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            logger.error("Another non-200-range status code was received")
            logger.error(e.status_code)
            logger.error(e.response)
            logger.error(e)
        return None

    @staticmethod
    def __to_chat_completion(merged_chunks: Dict[str, Any]) -> ChatCompletion:
        merged_chunks["object"] = "chat.completion"
        for choice in merged_chunks["choices"]:
            choice["message"] = choice["delta"]
            choice.pop("delta", None)
            choice.pop("logprobs", None)
        return ChatCompletion(**merged_chunks)

    @staticmethod
    def __merge_dicts(current: Dict[str, Any], delta: Dict[str, Any]) -> None:
        for key, value in delta.items():
            if value is None:
                continue
            elif key not in current:
                current[key] = value
            elif key in ["id", "model", "created", "index", "object", "system_fingerprint"]:
                continue
            else:
                if isinstance(value, str):
                    if current[key] is None:
                        current[key] = value
                    else:
                        current[key] += value
                elif isinstance(value, dict):
                    if current[key] is None:
                        current[key] = {}
                    GPTCompletion.__merge_dicts(current[key], value)
                elif isinstance(value, list):
                    if current[key] is None:
                        current[key] = []
                    for entry in value:
                        index = entry.get("index")
                        if index is not None:
                            existing_entry = next(
                                (e for e in current[key] if e.get("index") == index), None)
                            if existing_entry is None:
                                current[key].append(entry)
                            else:
                                GPTCompletion.__merge_dicts(
                                    existing_entry, entry)
