"""
OpenAI Client wrapper for LLM calls.
"""

import os
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }

            if max_tokens:
                request_params["max_completion_tokens"] = max_tokens

            if timeout:
                request_params["timeout"] = timeout

            request_params.update(kwargs)

            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content or ""

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")
