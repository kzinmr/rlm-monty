"""
Example of using RLM (REPL) with Monty sandbox on a needle-in-haystack problem.
"""

import random

from rlm import RLM_REPL


def generate_massive_context(num_lines: int = 100_000, answer: str = "1298418") -> str:
    print(f"Generating massive context with {num_lines} lines...")

    random_words = ["blah", "random", "text", "data", "content", "information", "sample"]

    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))

    magic_position = random.randint(int(num_lines * 0.4), int(num_lines * 0.6))
    lines[magic_position] = f"The magic number is {answer}"

    print(f"Magic number inserted at position {magic_position}")

    return "\n".join(lines)


def main():
    print("Example of using RLM (REPL) with Monty sandbox on a needle-in-haystack problem.")
    answer = str(random.randint(1000000, 9999999))
    context = generate_massive_context(num_lines=100_000, answer=answer)

    rlm = RLM_REPL(
        model="gpt-4o",
        recursive_model="gpt-4o-mini",
        enable_logging=True,
        max_iterations=10,
    )
    query = "I'm looking for a magic number. What is it?"
    result = rlm.completion(context=context, query=query)
    print(f"Result: {result}. Expected: {answer}")


if __name__ == "__main__":
    main()
