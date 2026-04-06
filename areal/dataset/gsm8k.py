from datasets import load_dataset


def get_gsm8k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_gsm8k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path=path, name="main", split=split)
    one_shot_suffix = (
        "\n\nAfter solving the above problem, please output your final answer in the following format:\n"
        "### The final answer is: $\\boxed{<your answer>}$\n"
        "Example:\n"
        "### The final answer is: $\\boxed{123}$\n"
        "The final answer should be given as precisely as possible (using LaTeX symbols such as \\sqrt, \\frac, \\pi, etc.). "
        "If the final answer involves a decimal approximation, it must be accurate to at least four decimal places."
    )

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["question"] + one_shot_suffix,
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
