import asyncio
import sys
import os

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)
sys.path.insert(0, parent_directory)


API_KEY = "YOUR API KEY HERE"


def callback(sentence: str, response_done: bool):
    """Callback function to handle responses from the GPT model.

    The library tries to segment the response into full sentences terminated by ASCII period (.)
    This works reasonably well for English. Might not be called at all if the model performs a 
    tool call that doesn't facilitate a response.

    Args:
        sentence: The full sentence parsed so far.
        response_done: true only on the last sentence of the response.

    """
    print(f"[{sentence}] response_done: {response_done}")


async def main():
    from pygptlink.gpt_context import GPTContext
    from pygptlink.gpt_completion import GPTCompletion

    context = GPTContext(model="gpt-3.5-turbo",
                         max_tokens=1000, max_response_tokens=100)
    completion = GPTCompletion(api_key=API_KEY, context=context)

    await completion.prompt(
        speaker="User", query="Hello World!", callback=callback)

    response = await completion.prompt(
        speaker="User", query="Is Finland a real country?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
