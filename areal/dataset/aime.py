from datasets import load_dataset


def _build_user_prompt(question: str) -> str:
    return question + "\nPlease put your final answer within \\boxed{}."


def get_aime_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    # Local json/jsonl files are loaded through the json builder.
    if path.endswith(".json") or path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files={split: path}, split=split)
    else:
        dataset = load_dataset(path=path, split=split)

    def process(sample):
        question = sample.get("question") or sample.get("problem")
        answer = sample.get("answer")
        if question is None:
            raise ValueError("AIME sample must contain `question` or `problem` field.")
        if answer is None:
            raise ValueError("AIME sample must contain `answer` field.")
        return {
            "messages": [{"role": "user", "content": _build_user_prompt(question)}],
            "answer": str(answer),
        }

    dataset = dataset.map(process)
    keep_columns = {"messages", "answer"}
    remove_columns = [c for c in dataset.column_names if c not in keep_columns]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    if max_length is not None:

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
