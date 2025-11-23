from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import csv
from copy import deepcopy
from tqdm import tqdm
import random
import math
import numpy as np
import pandas as pd
import os
import datasets
import argparse
from together import Together

# info about different models to simplify logit extraction
# at position n, we get probability of token at n + 1
model_info = {
    "google/gemma-2b-it": {"start_of_turn": "<start_of_turn>", "trigger": "model", "offset": 2, "context_window": 4000},
    "google/gemma-7b-it": {"start_of_turn": "<start_of_turn>", "trigger": "model", "offset": 2, "context_window": 4000},
    "google/gemma-1.1-2b-it": {"start_of_turn": "<start_of_turn>", "trigger": "model", "offset": 2, "context_window": 4000},
    "google/gemma-1.1-7b-it": {"start_of_turn": "<start_of_turn>", "trigger": "model", "offset": 2, "context_window": 4000},
    "google/gemma-2-2b-it": {"start_of_turn": "<start_of_turn>", "trigger": "model", "offset": 2, "context_window": 4000},
    "google/gemma-2-7b-it": {"start_of_turn": "<start_of_turn>", "trigger": "model", "offset": 2, "context_window": 4000},
    "Qwen/Qwen2-0.5B-Instruct": {"start_of_turn": "<|im_start|>", "trigger": "assistant", "offset": 2, "context_window": 16384},
    "Qwen/Qwen2-1.5B-Instruct": {"start_of_turn": "<|im_start|>", "trigger": "assistant", "offset": 2, "context_window": 16384},
    "meta-llama/Llama-3.2-1B-Instruct": {"start_of_turn": "<|start_header_id|>", "trigger": "assistant", "offset": 3, "context_window": 8000},
    "meta-llama/Llama-3.2-3B-Instruct": {"start_of_turn": "<|start_header_id|>", "trigger": "assistant", "offset": 3, "context_window": 8000},
    "meta-llama/Llama-3.1-8B-Instruct": {"start_of_turn": "<|start_header_id|>", "trigger": "assistant", "offset": 3, "context_window": 8000},
    "meta-llama/Llama-3.1-8B": {"start_of_turn": "<|start_header_id|>", "trigger": "assistant", "offset": 3, "context_window": 8000},
}

# together models
# at position n, we get probability of token at n
# e.g. '<|start_header_id|>', 'system', '<|end_header_id|>', '\n\n'
model_info_together = {
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {
        "start_of_turn": "<|start_header_id|>", "trigger": "assistant", "offset": 4, "context_window": 8192, "tokenizer": "meta-llama/Meta-Llama-3.1-405B-Instruct"
    },
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {
        "start_of_turn": "<|start_header_id|>", "trigger": "assistant", "offset": 4, "context_window": 8192, "tokenizer": "meta-llama/Meta-Llama-3.1-70B-Instruct"
    },
}

dataset_info = {
    "creak": {"end_pos": 1},
    "logiqa": {"end_pos": 1},
    "harmbench": {"end_pos": 3},
    "evals/persona/psychopathy": {"end_pos": 1},
    "evals/persona/machiavellianism": {"end_pos": 1},
    "evals/persona/narcissism": {"end_pos": 1},
}

tokeniser_mapping = {
    "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
}

# DATASET PREPROCESSING AND LOADING


def load_dataset(
    dataset: str = "harmbench",
    fixed_response: bool = False
):
    """Load and process the dataset for evaluation."""
    assert dataset in dataset_info, "Invalid dataset"
    qa_pairs = []
    if dataset == "harmbench":
        harmbench_prompts, harmbench_answers = [], []
        with open('datasets/harmbench/harmbench.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[4]:
                    harmbench_prompts.append(row[0] + '\n\n' + row[4])
                else:
                    harmbench_prompts.append(row[0])
        with open('datasets/harmbench/harmbench_wizardlm.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                harmbench_answers.append(
                    ('Here is how ' + row[0].split(' USER:')[0].strip().split(' AUSER:')[0].strip(),))
        qa_pairs = list(zip(harmbench_prompts, harmbench_answers))
    elif dataset == "logiqa":
        split = datasets.load_dataset("lucasmccabe/logiqa", split="test")

        def format_prompt(ex):
            prompt = f"Passage: {ex['context']}\nQuestion: {ex['query']}\nChoices:\n"
            for i, choice in enumerate(ex["options"]):
                letter = chr(ord("A") + i)
                prompt += f"{letter}. {choice}\n"
            prompt += "Answer:"
            answer = chr(ord("A") + ex["correct_option"])
            return (prompt, (answer,))
        qa_pairs = [format_prompt(ex) for ex in split]
    elif dataset == "creak":
        filename = "datasets/creak/dev.json"
        qa_pairs = []
        with open(filename, "r") as f:
            for line in f:
                data = json.loads(line)
                qa_pairs.append((data["sentence"], (data["label"],)))
    elif dataset.startswith("evals/persona/"):
        subset = dataset.split("/")[-1]
        filename = f"datasets/evals/persona/{subset}.jsonl"
        qa_pairs = []
        with open(filename, "r") as f:
            for line in f:
                data = json.loads(line)
                if fixed_response:
                    qa_pairs.append((data["question"], (" Yes", " No")))
                else:
                    qa_pairs.append(
                        (data["question"], (data["answer_matching_behavior"], data["answer_not_matching_behavior"])))
    return qa_pairs


def choose_answer_with_flips(
    dataset: str,
    answers: tuple[str, ...],
    hmm: int,
    flip_prob: float,
    rng: random.Random,
) -> str:
    """
    Choose an answer string for the given dataset/HMM, optionally flipping labels.
    Flipping only affects in-context answers; query answers remain the provided ones.
    """
    base_ans = answers[hmm]
    if flip_prob <= 0.0:
        return base_ans
    if rng.random() >= flip_prob:
        return base_ans

    if dataset.startswith("evals/persona/"):
        if len(answers) == 2:
            other_idx = 1 - hmm
            return answers[other_idx]
        return base_ans

    if dataset == "creak":
        label = base_ans.strip()
        lower = label.lower()
        if lower in ("true", "entailed"):
            flipped = "False" if lower == "true" else "not-entailed"
        elif lower in ("false", "not-entailed"):
            flipped = "True" if lower == "false" else "entailed"
        else:
            return base_ans
        prefix = " " if base_ans.startswith(" ") else ""
        return prefix + flipped

    if dataset == "logiqa":
        correct = base_ans.strip()
        choices = ["A", "B", "C", "D"]
        wrong_choices = [c for c in choices if c != correct]
        if not wrong_choices:
            return base_ans
        flipped = rng.choice(wrong_choices)
        prefix = " " if base_ans.startswith(" ") else ""
        return prefix + flipped

    return base_ans

# COMPUTING LOGITS


def format_token(tokenizer: AutoTokenizer, tok: int) -> str:
    """Format the token for some path patching experiment to show decoding diff"""
    return tokenizer.decode(tok).replace(" ", "_").replace("\n", "\\n")


def top_vals(tokenizer: AutoTokenizer, res: torch.Tensor, n: int = 10, return_results: bool = False) -> list | None:
    """Pretty print the top n values of a distribution over the vocabulary"""
    top_values, top_indices = torch.topk(res, n)
    ret = []
    for i, _ in enumerate(top_values):
        tok = format_token(tokenizer, top_indices[i].item())
        ret += [(tok, top_values[i].item())]
        if not return_results:
            print(f"{tok:<20} {top_values[i].item()}")
    if return_results:
        return ret


def print_gpu_stats():
    if torch.cuda.is_available():
        # Get the current GPU device
        gpu_id = torch.cuda.current_device()

        # Print the allocated and reserved memory
        print(
            f"Memory allocated on GPU {gpu_id}: {torch.cuda.memory_allocated(gpu_id) / 1024 ** 2:.2f} MB")
        print(
            f"Memory reserved on GPU {gpu_id}: {torch.cuda.memory_reserved(gpu_id) / 1024 ** 2:.2f} MB")
    else:
        print("No GPU available")


@torch.inference_mode()
def get_logits(
    tokens: torch.Tensor | list,
    logprobs: torch.Tensor | list,
    model_name: str = "google/gemma-2b-it",
    dataset: str = "harmbench",
    logits: bool = False,
    token_ids: torch.Tensor | list | None = None,
) -> list:
    # get model info
    info = None
    if model_name in model_info:
        info = model_info[model_name]
    elif model_name in model_info_together:
        info = model_info_together[model_name]
    else:
        raise ValueError("Model not supported")
    offset = dataset_info[dataset]["end_pos"]

    # find important positions
    data = []
    shots = 0
    for i, tok in enumerate(tokens):
        if tok == info["start_of_turn"] and tokens[i + 1] == info["trigger"]:
            start_pos = i + info["offset"]
            end_pos = i + info["offset"] + offset
            acc = None
            if logits:
                nll = 0
                for s in range(start_pos, end_pos):
                    lp = logprobs[s].log_softmax(dim=-1)
                    tgt = token_ids[s + 1]
                    nll -= lp[tgt]
                if offset == 1:
                    lp0 = logprobs[start_pos].log_softmax(dim=-1)
                    tgt0 = token_ids[start_pos + 1]
                    pred0 = lp0.argmax(dim=-1)
                    acc = float(pred0.item() == tgt0.item())
            else:
                nll = -sum(logprobs[start_pos:end_pos])
            if isinstance(nll, torch.Tensor):
                nll_value = nll.item()
            else:
                nll_value = float(nll)
            prob = math.exp(-nll_value)
            entry = {
                "shots": shots,
                "nll": nll_value,
                "prob": prob,
                "model": model_name,
            }
            if acc is not None:
                entry["acc"] = acc
            data.append(entry)
            shots += 1
    return data


# MODEL LOADING
device = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def load_model(model_name: str = "google/gemma-2b-it") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    with torch.inference_mode():
        torch.cuda.empty_cache()

        # load models
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device,
            # "attn_implementation": "flash_attention_2",
            # "load_in_8bit": True,
        }
        tokenizer_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name)  # same tokenizer necessary
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs)
        model.eval()
    return model, tokenizer


@torch.inference_mode()
def test_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    evals: list = ["evals/persona/psychopathy",
                   "evals/persona/machiavellianism", "evals/persona/narcissism"],
    ct: int = 50,
    shots: int = 1000,
    flip_prob: float = 0.0,
    seed: int = 42,
) -> list:
    model_name = model.config._name_or_path
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data = []
    for dataset in evals:
        qa_pairs = load_dataset(dataset)
        print(len(qa_pairs))
        for run_idx in tqdm(range(ct)):
            combined_seed = (int(seed) + hash(dataset) +
                             1000 * run_idx) & 0xFFFFFFFF
            rng = random.Random(combined_seed)
            random.shuffle(qa_pairs)
            question, answers = qa_pairs[0]
            for hmm in range(len(answers)):
                # construct the many-shot prompt
                chat = []
                for j in range(min(shots, len(qa_pairs))):
                    q_text = qa_pairs[j][0]
                    a_tuple = qa_pairs[j][1]
                    a_text = choose_answer_with_flips(
                        dataset=dataset,
                        answers=a_tuple,
                        hmm=hmm,
                        flip_prob=flip_prob,
                        rng=rng,
                    )
                    chat.append({"role": "user", "content": q_text})
                    chat.append({"role": "assistant", "content": a_text})
                    if model_name in tokeniser_mapping:
                        new_inputs = list(map(lambda turn: (
                            "User" if turn["role"] == "user" else "Assistant") + ": " + turn["content"], chat))
                        new_inputs = "\n\n".join(new_inputs)
                    else:
                        new_inputs = tokenizer.apply_chat_template(
                            chat, tokenize=False, add_generation_prompt=False)
                    new_inputs = tokenizer(new_inputs, return_tensors="pt")
                    if new_inputs["input_ids"].shape[-1] > model_info[model_name]["context_window"]:
                        break
                    inputs = new_inputs
                if run_idx == 0:
                    # store example for debugging
                    text = tokenizer.decode(inputs["input_ids"][0])
                    path = f"logs/real-lms/{model_name.replace('/', '_')}___{dataset.replace('/', '_')}___example.txt"
                    os.makedirs("logs/real-lms", exist_ok=True)
                    with open(path, "w+") as f:
                        f.write(text)

                # get logits and preprocess
                inputs.to(device)
                torch.cuda.empty_cache()
                logits = model(**inputs).logits[0].to("cpu")
                tokens = list(
                    map(lambda x: tokenizer.decode(x), inputs["input_ids"][0]))

                # now get the data
                more_data = get_logits(tokens, logits, model_name=model_name,
                                       dataset=dataset, logits=True, token_ids=inputs["input_ids"][0])
                for d in more_data:
                    d["hmm"] = hmm
                    d["dataset"] = dataset
                    d["flip_prob"] = float(flip_prob)
                data.extend(more_data)

            # empty cache after each forward pass
            torch.cuda.empty_cache()
    return data


def test_model_together(
    model_name: str,
    tokenizer: AutoTokenizer,
    evals: list = ["evals/persona/psychopathy",
                   "evals/persona/machiavellianism", "evals/persona/narcissism"],
    ct: int = 50,
    shots: int = 1000,
    flip_prob: float = 0.0,
    seed: int = 42,
) -> list:
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data = []
    for dataset in evals:
        # load dataset
        qa_pairs = load_dataset(dataset)
        print(len(qa_pairs))

        # construct prompt and get logprobs
        for run_idx in tqdm(range(ct)):
            combined_seed = (int(seed) + hash(dataset) +
                             1000 * run_idx) & 0xFFFFFFFF
            rng = random.Random(combined_seed)
            random.shuffle(qa_pairs)
            question, answers = qa_pairs[0]
            for hmm in range(len(answers)):
                # construct the many-shot prompt
                chat = []
                for j in range(min(shots, len(qa_pairs))):
                    q_text = qa_pairs[j][0]
                    a_tuple = qa_pairs[j][1]
                    a_text = choose_answer_with_flips(
                        dataset=dataset,
                        answers=a_tuple,
                        hmm=hmm,
                        flip_prob=flip_prob,
                        rng=rng,
                    )
                    chat.append({"role": "user", "content": q_text})
                    chat.append({"role": "assistant", "content": a_text})
                    new_inputs = tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=False)
                    new_inputs = tokenizer(new_inputs, return_tensors="pt")
                    if new_inputs["input_ids"].shape[-1] > model_info_together[model_name]["context_window"]:
                        chat = chat[:-2]
                        break

                # send to Together
                result = client.chat.completions.create(
                    model=model_name,
                    messages=chat,
                    stream=False,
                    max_tokens=1,
                    logprobs=1,
                    echo=True,
                )

                # extract logprobs
                tokens = result.prompt[0].logprobs.tokens
                logprobs = result.prompt[0].logprobs.token_logprobs

                # save example for debugging
                if run_idx == 0:
                    text = "".join(tokens)
                    path = f"logs/real-lms/{model_name.replace('/', '_')}___{dataset.replace('/', '_')}___example.txt"
                    with open(path, "w") as f:
                        f.write(text)

                # now get the data
                more_data = get_logits(
                    tokens, logprobs, model_name=model_name, dataset=dataset)
                for d in more_data:
                    d["hmm"] = hmm
                    d["dataset"] = dataset
                    d["flip_prob"] = float(flip_prob)
                data.extend(more_data)
    return data


@torch.inference_mode()
def main(
    model_name: str = "google/gemma-2b-it",
    datasets: list[str] | None = None,
    ct: int = 50,
    shots: int = 1000,
    flip_prob: float = 0.0,
    seed: int = 42,
):
    evals = datasets or list(dataset_info.keys())
    models = [model_name]
    if model_name == "all":
        models = list(model_info.keys())
    os.makedirs("logs/real-lms", exist_ok=True)
    flip_str = f"flip{flip_prob:.2f}".replace(".", "p")
    for model_name in models:
        model, tokenizer = None, None
        print(f"Running: {model_name}")
        for dataset in evals:
            # skip if already computed
            m = model_name.replace("/", "_")
            d = dataset.replace("/", "_")
            path = f"logs/real-lms/{m}___{d}___{flip_str}___means.csv"
            if os.path.exists(path):
                continue
            print(f"Running: {model_name} on {dataset}")

            # load model and collect data
            if model_name in model_info:
                if model is None:
                    model, tokenizer = load_model(model_name)
                    print("torch.cuda.memory_allocated: %f" %
                          (torch.cuda.memory_allocated(0)))
                    print("torch.cuda.memory_reserved: %f" %
                          (torch.cuda.memory_reserved(0)))
                    print("torch.cuda.max_memory_reserved: %f" %
                          (torch.cuda.max_memory_reserved(0)))
                data = test_model(
                    model,
                    tokenizer,
                    evals=[dataset],
                    ct=ct,
                    shots=shots,
                    flip_prob=flip_prob,
                    seed=seed,
                )
            elif model_name in model_info_together:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_info_together[model_name]["tokenizer"])
                data = test_model_together(
                    model_name,
                    tokenizer,
                    evals=[dataset],
                    ct=ct,
                    shots=shots,
                    flip_prob=flip_prob,
                    seed=seed,
                )
            else:
                raise ValueError(f"model {model_name} not found in configs")

            # assemble data
            df = pd.DataFrame(data)
            print(df)
            df["prob"] = df["nll"].map(lambda x: math.exp(-x))
            df["shots"] += 1
            print(len(df))

            # make NLL the actual log expected prob
            df_mean = df.groupby(
                ["dataset", "shots", "model", "hmm", "flip_prob"], as_index=False
            ).mean(numeric_only=True)
            df_mean["nll_avg"] = df_mean["nll"]
            df_mean["nll"] = df_mean["prob"].map(lambda x: -math.log(x))
            df_mean.to_csv(path, index=False)
            raw_path = f"logs/real-lms/{m}___{d}___{flip_str}___raw.csv"
            df.to_csv(raw_path, index=False)
            print("Saved to: ", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2b-it",
        help="HF model name or key in model_info / model_info_together.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of datasets from dataset_info to run. If None, runs all datasets in dataset_info.",
    )
    parser.add_argument(
        "--ct",
        type=int,
        default=50,
        help="Number of random shuffles / prompts per dataset.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1000,
        help="Maximum number of in-context demonstrations per prompt.",
    )
    parser.add_argument(
        "--flip_prob",
        type=float,
        default=0.0,
        help="Probability of flipping each in-context answer label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for RNGs controlling flips and shuffling.",
    )
    args = parser.parse_args()

    main(
        model_name=args.model,
        datasets=args.datasets,
        ct=args.ct,
        shots=args.shots,
        flip_prob=args.flip_prob,
        seed=args.seed,
    )
