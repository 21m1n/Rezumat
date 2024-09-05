import tiktoken
import anthropic


# estimating the cost
def count_tokens(input_string: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(input_string)
    return len(tokens)


def calculate_cost(input_string: str, cost_per_million_tokens: float = 5) -> float:
    num_tokens = count_tokens(input_string)
    total_cost = (num_tokens / 1_000_000) * cost_per_million_tokens
    return total_cost


# client = anthropic.Client()
# token_count = client.count_tokens(complete_template)
# print(token_count)
# generate unique id
