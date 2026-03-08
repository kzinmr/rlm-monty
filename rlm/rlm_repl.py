"""
Simple Recursive Language Model (RLM) with REPL environment using Monty sandbox.
"""

from typing import Any, Dict, List, Optional

from rlm import RLM
from rlm.repl import REPLEnv
from rlm.utils.llm import OpenAIClient
from rlm.utils.prompts import DEFAULT_QUERY, build_system_prompt, next_action_prompt
import rlm.utils.utils as utils

from rlm.logger.root_logger import ColorfulLogger
from rlm.logger.repl_logger import REPLEnvLogger


class RLM_REPL(RLM):
    """
    LLM Client that can handle long contexts by recursively calling itself.
    Uses Monty sandbox for secure code execution.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        recursive_model: str = "gpt-4o-mini",
        max_iterations: int = 20,
        depth: int = 0,
        enable_logging: bool = False,
    ):
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.llm = OpenAIClient(api_key, model)

        self.repl_env: REPLEnv | None = None
        self.depth = depth
        self._max_iterations = max_iterations

        self.logger = ColorfulLogger(enabled=enable_logging)
        self.repl_env_logger = REPLEnvLogger(enabled=enable_logging)

        self.messages: List[Dict[str, str]] = []
        self.query: str | None = None

    def setup_context(
        self,
        context: List[str] | str | List[Dict[str, str]],
        query: Optional[str] = None,
    ):
        """
        Setup the context for the RLMClient.

        Args:
            context: The large context to analyze in the form of a list of messages, string, or Dict
            query: The user's question
        """
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.logger.log_query_start(query)

        self.messages = build_system_prompt()
        self.logger.log_initial_messages(self.messages)

        context_data, context_str = utils.convert_context_for_repl(context)

        self.repl_env = REPLEnv(
            context_json=context_data,
            context_str=context_str,
            recursive_model=self.recursive_model,
        )

        return self.messages

    def completion(
        self,
        context: List[str] | str | List[Dict[str, str]],
        query: Optional[str] = None,
    ) -> str:
        """
        Given a query and a (potentially long) context, recursively call the LM
        to explore the context and provide an answer using a REPL environment.
        """
        self.messages = self.setup_context(context, query)

        iteration = 0
        for iteration in range(self._max_iterations):
            response = self.llm.completion(self.messages + [next_action_prompt(self.query or "", iteration)])

            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=code_blocks is not None)

            if code_blocks is not None:
                self.messages = utils.process_code_execution(
                    response,
                    self.messages,
                    self.repl_env,
                    self.repl_env_logger,
                    self.logger,
                )
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": "You responded with:\n" + response,
                }
                self.messages.append(assistant_message)

            final_answer = utils.check_for_final_answer(response, self.repl_env, self.logger)

            if final_answer:
                self.logger.log_final_response(final_answer)
                return final_answer

        print("No final answer found in any iteration")
        self.messages.append(next_action_prompt(self.query or "", iteration, final_answer=True))
        final_answer = self.llm.completion(self.messages)
        self.logger.log_final_response(final_answer)

        return final_answer

    def cost_summary(self) -> Dict[str, Any]:
        """Get the cost summary of the Root LM + Sub-RLM Calls."""
        raise NotImplementedError("Cost tracking not implemented for RLM REPL.")

    def reset(self):
        """Reset the (REPL) environment and message history."""
        self.repl_env = REPLEnv(recursive_model=self.recursive_model)
        self.messages = []
        self.query = None


if __name__ == "__main__":
    pass
