import inspect
import re
from enum import Enum, auto
from typing import List, Optional

from pygptlink.gpt_tool_definition import GPTToolDefinition
from pygptlink.gpt_logging import logger


class DocSection(Enum):
    SUMMARY = auto()
    ARGS = auto()
    RETURNS = auto()
    RAISES = auto()


# This class encapsulates all the tools available for the GPT model to use and can generate
# automatic descriptions for all tools suitable for use in completions.
class GPTTools:

    @staticmethod
    def _validate_type(type: str):
        # Optionally map python types to JSON Schema
        # https://json-schema.org/understanding-json-schema
        if re.match("List\[.*?\]", type):
            type = "array"

        mapping = {'str': 'string', 'int': 'integer',
                   'bool': 'boolean', 'float': 'number'}
        type = type.strip()
        type = mapping.get(type, type)
        if type in {'string', 'integer', 'number', 'boolean', 'array'}:
            return type
        raise RuntimeError(f"Unknown parameter type: {type} encountered!")

    def _describe_methods(self) -> List[GPTToolDefinition]:
        ans = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            doc_string = inspect.getdoc(method)
            if not doc_string:
                continue
            if name[0] == "_":
                continue  # Skip "protected/private" methods

            doc_lines = doc_string.splitlines()
            previous_blank = False
            current_section = DocSection.SUMMARY

            summary = ""
            required_args = []
            optional_args = []
            current_arg = {}

            for line in doc_lines:
                if "args:".casefold() == line.casefold() and previous_blank:
                    current_section = DocSection.ARGS
                elif "returns:".casefold() == line.casefold() and previous_blank:
                    current_section = DocSection.RETURNS
                elif "raises:".casefold() == line.casefold() and previous_blank:
                    current_section = DocSection.RAISES
                elif line:
                    if current_section == DocSection.SUMMARY:
                        if summary:
                            summary += " " + line.strip()
                        else:
                            summary = line.strip()
                    elif current_section == DocSection.ARGS:
                        match = re.match(
                            pattern=r"\s*(\w+) \(([^)]+)\): (.+)", string=line)
                        if match:
                            # New argument found
                            current_arg = {}
                            current_arg["name"] = match.group(1).strip()
                            current_arg["description"] = match.group(3).strip()

                            opt_match = re.match(
                                pattern=r"Optional\[(\w+)\].*|(.*?), optional.*", string=match.group(2).strip())
                            if opt_match:
                                current_arg["type"] = GPTTools._validate_type(
                                    opt_match.group(1) or opt_match.group(2))
                                optional_args.append(current_arg)
                            else:
                                current_arg["type"] = GPTTools._validate_type(
                                    match.group(2).strip())
                                required_args.append(current_arg)
                        else:
                            if not current_arg:
                                # This is a bug at startup, crash early.
                                raise RuntimeError("Current arg is empty!")
                            # Continuation of previous arg's description
                            current_arg["description"] += " " + line.strip()
                        pass
                    elif current_section == DocSection.RETURNS:
                        # Ignore the returns section, we don't need it
                        pass
                    elif current_section == DocSection.RAISES:
                        # Ignore the raises section, we don't need it
                        pass
                    else:
                        # This is a bug at startup, crash early.
                        raise RuntimeError("Unknown enum, this is a bug")
                previous_blank = not line

            tool = GPTToolDefinition(name=name, callback=method, description=summary,
                                     required_args=required_args, optional_args=optional_args)
            logger.debug(f"Parsed tool: {tool.__dict__}")
            ans.append(tool)
        return ans

    def exclude_me(self):
        pass

    def test(self, name: str, age: int, time: Optional[float] = None) -> str:
        """Load bearing docstring.
        Second line of summary.

        Args:
            name (str): _description_ 
            continued on a new line
            age (int): _description_
            time (Optional[float], optional): _description_. Defaults to None.

        Returns:
            str: _description_
        """
        return ""
