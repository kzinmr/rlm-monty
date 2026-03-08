"""Utility functions for the RLM REPL Client."""

import re
from typing import Any, Dict, List, Optional, Tuple


def find_code_blocks(text: str) -> List[str] | None:
    """
    Find REPL code blocks in text wrapped in triple backticks and return List of content(s).
    Returns None if no code blocks are found.
    """
    pattern = r"```repl\s*\n(.*?)\n```"
    results: List[str] = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results if results else None


def find_final_answer(text: str) -> Optional[Tuple[str, str]]:
    """
    Find FINAL(...) or FINAL_VAR(...) statement in response and return (type, content).
    Returns None if neither pattern is found.
    """
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ("FINAL_VAR", match.group(1).strip())

    final_pattern = r"^\s*FINAL\((.*?)\)"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ("FINAL", match.group(1).strip())

    return None


def add_execution_result_to_messages(
    messages: List[Dict[str, str]],
    code: str,
    result: str,
    max_character_length: int = 100000,
) -> List[Dict[str, str]]:
    """
    Add code execution result to the conversation messages.
    """
    if len(result) > max_character_length:
        result = result[:max_character_length] + "..."

    execution_message = {
        "role": "user",
        "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
    }
    messages.append(execution_message)
    return messages


def format_execution_result(
    stdout: str,
    stderr: str,
    locals_dict: Dict[str, Any],
    truncate_length: int = 100,
) -> str:
    """
    Format the execution result as a string for display.
    """
    result_parts: List[str] = []

    if stdout:
        result_parts.append(f"\n{stdout}")

    if stderr:
        result_parts.append(f"\n{stderr}")

    important_vars: Dict[str, str] = {}
    for key, value in locals_dict.items():
        if not key.startswith("_") and key not in [
            "__builtins__",
            "__name__",
            "__doc__",
        ]:
            try:
                if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                    if isinstance(value, str) and len(value) > truncate_length:
                        important_vars[key] = f"'{value[:truncate_length]}...'"
                    else:
                        important_vars[key] = repr(value)
            except Exception:
                important_vars[key] = f"<{type(value).__name__}>"

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


def execute_code(repl_env, code: str, repl_env_logger, logger) -> str:
    """
    Execute code in the REPL environment and return formatted result.
    """
    try:
        result = repl_env.code_execution(code)

        formatted_result = format_execution_result(result.stdout, result.stderr, result.locals)
        repl_env_logger.log_execution(code, result.stdout, result.stderr, result.execution_time)
        repl_env_logger.display_last()

        logger.log_tool_execution("CODE_EXECUTION", formatted_result)

        return formatted_result

    except Exception as e:
        return f"Error executing code: {str(e)}"


def process_code_execution(
    response: str,
    messages: List[Dict[str, str]],
    repl_env,
    repl_env_logger,
    logger,
) -> List[Dict[str, str]]:
    """
    Process code execution from the model response.
    """
    code_blocks = find_code_blocks(response)

    if code_blocks:
        for code in code_blocks:
            execution_result = execute_code(repl_env, code, repl_env_logger, logger)

            messages = add_execution_result_to_messages(messages, code, execution_result)

    return messages


def check_for_final_answer(response: str, repl_env, logger) -> Optional[str]:
    """Check if response contains a final answer."""
    result = find_final_answer(response)
    if result is None:
        return None

    answer_type, content = result

    if answer_type == "FINAL":
        return content
    elif answer_type == "FINAL_VAR":
        try:
            variable_name = content.strip().strip('"').strip("'").strip("\n").strip("\r")

            if repl_env is not None and variable_name in repl_env.locals:
                variable_value = repl_env.locals[variable_name]
                return str(variable_value)
            else:
                error_msg = f"Variable '{variable_name}' not found in REPL environment"
                logger.log_tool_execution("FINAL_VAR", error_msg)
                return None
        except Exception as e:
            error_msg = f"Error retrieving variable '{variable_name}': {str(e)}"
            logger.log_tool_execution("FINAL_VAR", error_msg)
            return None

    return None


def convert_context_for_repl(
    context: Any,
) -> Tuple[Dict[str, Any] | List[Any] | None, str | None]:
    """
    Convert REPL context to appropriate format.
    """
    if isinstance(context, dict):
        return context, None
    elif isinstance(context, str):
        return None, context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            return context_data, None
        else:
            return context, None
    else:
        return context, None
