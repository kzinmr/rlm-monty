"""
Example prompt templates for the RLM REPL Client.
"""

from typing import Dict, List

DEFAULT_QUERY = (
    "Please read through the context and answer any queries or respond to any instructions contained within it."
)

REPL_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example:
```repl
chunk = context[:10000]
print(chunk)
```

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""


def build_system_prompt() -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": REPL_SYSTEM_PROMPT,
        },
    ]


USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original query: "{query}".

Continue using the REPL environment, which has the `context` variable, and determine your answer. Your next action:"""


def next_action_prompt(query: str, iteration: int = 0, final_answer: bool = False) -> Dict[str, str]:
    if final_answer:
        return {
            "role": "user",
            "content": "Based on all the information you have, provide a final answer to the user's query.",
        }
    if iteration == 0:
        safeguard = "You have not interacted with the REPL environment or seen your context yet. Your next action should be to look through, don't just provide a final answer yet.\n\n"
        return {"role": "user", "content": safeguard + USER_PROMPT.format(query=query)}
    else:
        return {
            "role": "user",
            "content": "The history before is your previous interactions with the REPL environment. "
            + USER_PROMPT.format(query=query),
        }
