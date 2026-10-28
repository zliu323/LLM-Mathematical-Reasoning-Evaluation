import argparse
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import pandas as pd
import json
import re
import os
import gc
from fractions import Fraction


# model
MODELS = [
    "Qwen/Qwen2.5-Math-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
]

# 3 prompt methods
PROMPT_METHODS = ["direct", "cot", "self_refine"]



MAX_SAMPLES_PER_DATASET = None
MAX_NEW_TOKENS = 4000
SEED = 42
DO_SAMPLE = False

# Save results next to this Python script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# prompts
def build_prompt(question, method, previous_answer=None, feedback=None):
    if method == "direct":
        return f"""You are solving a mathematics problem.
output only:
Final answer: <number or expression>

Problem:
{question}
"""

    elif method == "cot":
        return f"""You are solving a mathematics problem.
Reason step by step, showing the key intermediate steps clearly.
Do not skip important calculations.

On the last line, output only:
Final answer: <number or expression>

Problem:
{question}
"""

    elif method == "self_refine_initial":
        return f"""Solve the following mathematics problem carefully step by step.
On the last line, output only:
Final answer: <number or expression>

Problem:
{question}
"""

    elif method == "self_refine_feedback":
        return f"""You are reviewing the solution to a mathematics problem.
Check the steps and final answer carefully. If it is wrong: explain briefly how to fix it
If the solution is correct, say "The solution appears correct."
Do not solve the whole problem again.

Problem:
{question}

Previous solution:
{previous_answer}
"""

    elif method == "self_refine_final":
        return f"""Solve the problem again using the feedback.
End your response with exactly:
Final Answer: <number or expression>

Problem:
{question}

Previous solution:
{previous_answer}

Feedback:
{feedback}
"""

    else:
        raise ValueError(f"Unknown method: {method}")


# GENERATION
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=None
):
    max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1
    }

    if DO_SAMPLE:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        })
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    return text


# ANSWER PROCESSING
def extract_boxed_content(text):
    start = text.rfind(r"\boxed{")
    if start == -1:
        return None

    i = start + len(r"\boxed{")
    depth = 1
    content = []

    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            content.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                content.append(ch)
        else:
            content.append(ch)
        i += 1

    if depth == 0:
        return "".join(content).strip()
    return None


def extract_final_answer(text):
    text = str(text).strip()

    # 1. Try boxed answer first
    boxed = extract_boxed_content(text)
    if boxed:
        return boxed.strip().strip(".")

    # 2. Try explicit final-answer formats
    patterns = [
        r"Final answer\s*:\s*([^\n]+)",
        r"Answer\s*:\s*([^\n]+)"
    ]

    matches = [
        match
        for pattern in patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]
    if matches:
        match = max(matches, key=lambda item: item.start())
        return match.group(1).strip().strip(".")

    # 3. Fallback: last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(ans):
    ans = str(ans).strip().lower()

    # remove common latex wrappers / spacing
    if ans.startswith("\\boxed{") and ans.endswith("}"):
        ans = ans[len("\\boxed{"):-1]
    ans = ans.replace("\\left", "")
    ans = ans.replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\tfrac", "\\frac")
    ans = ans.replace("$", "")
    ans = ans.replace(" ", "")
    ans = re.sub(r"\\text\{([^{}]*)\}", r"\1", ans)

    ans = ans.strip()

    # turn simple decimals like 3.0 into 3
    ans = re.sub(r'(?<![\d])(\d+)\.0(?!\d)', r'\1', ans)

    return ans


def parse_numeric_answer(ans):
    ans = normalize_answer(ans)
    ans = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", ans)

    frac_match = re.fullmatch(r'(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}', ans)
    if frac_match:
        sign = -1 if frac_match.group(1) == "-" else 1
        return sign * Fraction(int(frac_match.group(2)), int(frac_match.group(3)))

    if re.fullmatch(r'-?\d+', ans):
        return Fraction(int(ans), 1)

    if re.fullmatch(r'-?\d+\.\d+', ans):
        return Fraction(ans)

    return None


def split_top_level_commas(ans):
    parts = []
    current = []
    depth = 0

    for ch in ans:
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth -= 1
        current.append(ch)

    parts.append("".join(current))
    return [part for part in parts if part]


def is_correct(pred, gold):
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    pred_num = parse_numeric_answer(pred)
    gold_num = parse_numeric_answer(gold)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num

    if (
        pred_norm.startswith("(") and pred_norm.endswith(")")
        and gold_norm.startswith("(") and gold_norm.endswith(")")
    ):
        pred_parts = split_top_level_commas(pred_norm[1:-1])
        gold_parts = split_top_level_commas(gold_norm[1:-1])
        if len(pred_parts) == len(gold_parts):
            return all(is_correct(p, g) for p, g in zip(pred_parts, gold_parts))

    return False


def response_length_words(text):
    return len(text.split())


def extract_gsm8k_gold(answer_text):
    answer_text = str(answer_text)
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


# DATASETS
def load_all_datasets(max_samples, dataset_names=None):
    dataset_names = dataset_names or ["gsm8k_test", "math500", "aime2024"]
    loaded = {}

    if "gsm8k_test" in dataset_names:
        print("Loading GSM8K test...")
        gsm8k = load_dataset("gsm8k", "main", split="test")
        if max_samples is not None:
            gsm8k = gsm8k.select(range(min(max_samples, len(gsm8k))))
        loaded["gsm8k_test"] = gsm8k

    if "math500" in dataset_names:
        print("Loading MATH-500 test...")
        math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
        if max_samples is not None:
            math500 = math500.select(range(min(max_samples, len(math500))))
        loaded["math500"] = math500

    if "aime2024" in dataset_names:
        print("Loading AIME 2024...")
        aime2024 = load_dataset("HuggingFaceH4/aime_2024", split="train")
        if max_samples is not None:
            aime2024 = aime2024.select(range(min(max_samples, len(aime2024))))
        loaded["aime2024"] = aime2024

    return loaded


def get_question_and_gold(dataset_name, sample):
    if dataset_name == "gsm8k_test":
        question = sample["question"]
        gold = extract_gsm8k_gold(sample["answer"])
    elif dataset_name == "math500":
        question = sample["problem"]
        gold = str(sample["answer"]).strip()
    elif dataset_name == "aime2024":
        question = sample["problem"]
        gold = str(sample["answer"]).strip()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return question, gold


# METHODS
def run_direct(model, tokenizer, question):
    prompt = build_prompt(question, "direct")
    return generate_text(model, tokenizer, prompt)


def run_cot(model, tokenizer, question):
    prompt = build_prompt(question, "cot")
    return generate_text(model, tokenizer, prompt)


def run_self_refine(model, tokenizer, question):
    initial_prompt = build_prompt(question, "self_refine_initial")
    initial_output = generate_text(model, tokenizer, initial_prompt)

    feedback_prompt = build_prompt(
        question,
        "self_refine_feedback",
        previous_answer=initial_output
    )
    feedback_output = generate_text(model, tokenizer, feedback_prompt)

    final_prompt = build_prompt(
        question,
        "self_refine_final",
        previous_answer=initial_output,
        feedback=feedback_output
    )
    final_output = generate_text(model, tokenizer, final_prompt)

    return initial_output, feedback_output, final_output

# RUN ONE MODEL
def run_one_model(model_name, all_datasets):
    print("\n" + "=" * 80)
    print(f"Loading model: {model_name}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto"
    )
    model.eval()

    summary_rows = []

    for dataset_name, dataset in all_datasets.items():
        print(f"\n--- Dataset: {dataset_name} | Samples: {len(dataset)} ---")

        for method in PROMPT_METHODS:
            print(f"\nRunning method: {method}")

            detailed_results = []
            correct_count = 0
            total_length = 0

            for idx, sample in enumerate(dataset):
                question, gold = get_question_and_gold(dataset_name, sample)

                print(f"  [{idx + 1}/{len(dataset)}] processing...")

                row = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "method": method,
                    "sample_index": idx,
                    "question": question,
                    "gold_answer": gold
                }

                if method == "direct":
                    raw_output = run_direct(model, tokenizer, question)
                    pred = extract_final_answer(raw_output)
                    length_words = response_length_words(raw_output)

                    row["raw_output"] = raw_output
                    row["prediction"] = pred
                    row["response_length_words"] = length_words

                elif method == "cot":
                    raw_output = run_cot(model, tokenizer, question)
                    pred = extract_final_answer(raw_output)
                    length_words = response_length_words(raw_output)

                    row["raw_output"] = raw_output
                    row["prediction"] = pred
                    row["response_length_words"] = length_words

                elif method == "self_refine":
                    initial_output, feedback_output, final_output = run_self_refine(
                        model, tokenizer, question
                    )
                    pred = extract_final_answer(final_output)

                    total_text = initial_output + "\n" + feedback_output + "\n" + final_output
                    length_words = response_length_words(total_text)

                    row["initial_output"] = initial_output
                    row["feedback_output"] = feedback_output
                    row["raw_output"] = final_output
                    row["prediction"] = pred
                    row["response_length_words"] = length_words

                else:
                    raise ValueError(f"Unknown method: {method}")

                correct = is_correct(row["prediction"], gold)
                row["correct"] = correct

                correct_count += int(correct)
                total_length += row["response_length_words"]

                detailed_results.append(row)

            accuracy = correct_count / len(dataset) if len(dataset) > 0 else 0.0
            avg_len = total_length / len(dataset) if len(dataset) > 0 else 0.0

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Average response length: {avg_len:.2f} words")

            safe_model_name = model_name.replace("/", "_")
            json_path = os.path.join(
                RESULT_DIR,
                f"{safe_model_name}_{dataset_name}_{method}.json"
            )

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)

            summary_rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "method": method,
                "accuracy": accuracy,
                "avg_response_length_words": avg_len
            })

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary_rows


# MAIN
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate open-source language models on mathematical reasoning benchmarks."
    )
    parser.add_argument("--models", nargs="+", default=MODELS, help="Hugging Face model IDs")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["gsm8k_test", "math500", "aime2024"],
        default=["gsm8k_test", "math500", "aime2024"],
    )
    parser.add_argument(
        "--methods", nargs="+", choices=PROMPT_METHODS, default=PROMPT_METHODS
    )
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_PER_DATASET)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use temperature sampling instead of deterministic greedy decoding",
    )
    parser.add_argument("--output-dir", default=RESULT_DIR)
    return parser.parse_args()


def main():
    global MAX_NEW_TOKENS, DO_SAMPLE, RESULT_DIR, PROMPT_METHODS

    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be a positive integer")

    MAX_NEW_TOKENS = args.max_new_tokens
    DO_SAMPLE = args.sample
    RESULT_DIR = os.path.abspath(args.output_dir)
    PROMPT_METHODS = args.methods
    os.makedirs(RESULT_DIR, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Results will be saved to: {RESULT_DIR}")
    print(f"Decoding: {'sampling' if DO_SAMPLE else 'greedy'} | seed={args.seed}")
    print("Loading all datasets...")
    all_datasets = load_all_datasets(args.max_samples, args.datasets)

    all_summary_rows = []

    for model_name in args.models:
        rows = run_one_model(model_name, all_datasets)
        all_summary_rows.extend(rows)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_csv_path = os.path.join(RESULT_DIR, "summary_all.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print("\nFinished all experiments.")
    print(f"Summary saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()
