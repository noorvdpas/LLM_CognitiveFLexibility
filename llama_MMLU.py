# Feedback from Nas:
# AIM: try to reduce the time taken to run MMLU
# - Probably would run a lot faster if you construct a dataset or list of questions to be processed in parallel # - Maybe also look at whether you can run the context once into the LLM and then save the state (don't spend too long on this)
# # AIM: We want to know that improvements are not just random chance
# - Make sure the outputs are deterministic (i.e. that you get exactly the same response if you re-run the model)
# - Save the random seed you are using to select questions #
# AIM: Have a clean structure for loading contexts (and testing them)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from datasets import load_dataset
import csv
import re
import random
import time
from datetime import datetime
import os
import fcntl
from tqdm import tqdm
import torch
import argparse

# Set permanent cache directories
CACHE_DIR = "/scratch/fast/huggingface_cache"
DATASETS_CACHE_DIR = "/scratch/fast/huggingface_datasets_cache"

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run MMLU evaluation with contexts on specified CUDA device."
)
parser.add_argument(
    "--chunk",
    type=str,
    default="1/1",
    help="Chunk to process in format 'current/total' (e.g., '1/5' for first of 5 chunks, default: '1/1')",
)
parser.add_argument(
    "--device", type=int, default=0, help="CUDA device ID to use (default: 0)"
)
parser.add_argument(
    "--contexts_file",
    type=str,
    default="contexts_wikipedia.csv",
    choices=[
        "contexts_wikipedia.csv",
        "contexts_conspiracy.csv",
        "contexts_isot_fake.csv",
        "contexts_isot_true.csv",
    ],
    help="Contexts file to use (default: contexts_wikipedia.csv)",
)
args = parser.parse_args()

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

# Ensure cache directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATASETS_CACHE_DIR, exist_ok=True)

# Set environment variables for permanent caching
os.environ["HF_DATASETS_CACHE"] = DATASETS_CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR


# Pre-download and cache the MMLU dataset permanently
def cache_mmlu_dataset():
    """Pre-download and cache the MMLU dataset to avoid repeated downloads."""
    try:
        print("Checking/caching MMLU dataset...")
        load_dataset("cais/mmlu", "all", split="test", cache_dir=DATASETS_CACHE_DIR)
        print("MMLU dataset cached successfully!")
    except Exception as e:
        print(f"Warning: Could not cache dataset: {e}")


# Cache the dataset (run once to ensure permanent caching)
cache_mmlu_dataset()

start = time.time()

# model_name = "meta-llama/Llama-3.1-8B"
# model_name = "Qwen/Qwen2.5-7B"
# model_name = "google/gemma-2-9b"
model_name = "Qwen/Qwen3-8B"
test_name = "cais/mmlu"

CACHE_DIR = "/scratch/fast/huggingface_cache"
MAX_TOKENS = 5
NUM_TEST = 1000
BATCH_SIZE = 4  # Reduced to avoid OOM, increase gradually if possible
SEED = 28

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
tokenizer.padding_side = "left"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    cache_dir=CACHE_DIR,
)

tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id

subjects = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

all_data = []
print("Loading MMLU dataset...")
ds = load_dataset("cais/mmlu", "all", split="test", cache_dir=DATASETS_CACHE_DIR)
print(f"Dataset loaded with {len(ds)} total examples")

for subject in tqdm(subjects, desc="Filtering subjects"):
    subject_data = ds.filter(lambda x: x["subject"] == subject)
    all_data.extend(subject_data)


# dataset = all_data

random.seed(SEED)
# random.shuffle(all_data)
dataset = random.sample(all_data, NUM_TEST)
# dataset = all_data


def format_prompt_mmlu(question: str, choices: list[str]) -> str:
    formatted_choices = []
    for i, possible_answer in enumerate(choices):
        answer_letter = chr(65 + i)
        formatted_choice = f"{answer_letter}. {possible_answer}"
        formatted_choices.append(formatted_choice)
    choices_str = "\n".join(formatted_choices)
    return (
        f"Choose the correct answer from the options below. "
        f"Answer ONLY with a single letter (A, B, C, or D):\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choices_str}\n\n"
        "Answer:"
    )


def format_prompt_with_context(
    context_text: str, question: str, choices: list[str]
) -> str:
    formatted_choices = []
    for i, possible_answer in enumerate(choices):
        answer_letter = chr(65 + i)
        formatted_choice = f"{answer_letter}. {possible_answer}"
        formatted_choices.append(formatted_choice)
    choices_str = "\n".join(formatted_choices)
    return (
        f"Read this coming text very carefully: \n\n"
        f"{context_text}\n\n"
        f"Okay, and now, with all of that in mind, for the following question. Answer ONLY with a single letter (A, B, C, or D):\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choices_str}\n\n"
        "Answer ONLY with a single letter (A, B, C, or D) \n"
        "Answer:"
    )


def extract_mc_answer(output_text: str) -> str:
    text = output_text.upper()
    match = re.search(r"ANSWER:\s*([A-D])", text)
    if match:
        return match.group(1)
    return ""


contexts_file = args.contexts_file
contexts = {}
with open(contexts_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        title = row["context_title"]
        ctype = row["context_type"]
        text = row["context_text"]
        if title not in contexts:
            contexts[title] = {}
        contexts[title][ctype] = text

# titles_to_test = ["Astar_(game)"]
titles_to_test = list(contexts.keys())

# Sort titles for consistent chunking
titles_to_test.sort()

# Parse chunk argument
chunk_parts = args.chunk.split("/")
current_chunk = int(chunk_parts[0]) - 1  # 0-based index
total_chunks = int(chunk_parts[1])

# Calculate chunk range
total_titles = len(titles_to_test)
chunk_size = total_titles // total_chunks
start_idx = current_chunk * chunk_size
end_idx = start_idx + chunk_size if current_chunk < total_chunks - 1 else total_titles

# Select the chunk
titles_to_test = titles_to_test[start_idx:end_idx]

print(f"Titles to test: {titles_to_test}")
print(
    f"Processing chunk {args.chunk}: {len(titles_to_test)} contexts (indices {start_idx} to {end_idx-1}) out of {total_titles} total on device {args.device}"
)

context_types = ["clean", "meaningful_shuffle"]

if args.contexts_file == "contexts_conspiracy.csv":
    csv_results = "results_conspiracy.csv"
elif args.contexts_file == "contexts_wikipedia.csv":
    csv_results = "results_wikipedia.csv"
elif args.contexts_file == "contexts_isot_fake.csv":
    csv_results = "results_isot_fake.csv"
elif args.contexts_file == "contexts_isot_true.csv":
    csv_results = "results_isot_true.csv"
file_exists = os.path.exists(csv_results)

# Load existing results to avoid re-processing
existing_results = {}
if file_exists:
    with open(csv_results, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row["context_title"]
            ctype = row["context_type"]
            model_name_csv = row["model_name"]
            if title not in existing_results:
                existing_results[title] = {}
            if ctype not in existing_results[title]:
                existing_results[title][ctype] = {}
            existing_results[title][ctype][model_name_csv] = float(row["accuracy"])

accuracy_dict = {
    title: {ctype: 0.0 for ctype in context_types} for title in titles_to_test
}

# Populate accuracy_dict with existing results for this model
for title in titles_to_test:
    if title in existing_results:
        for ctype in context_types:
            if (
                ctype in existing_results[title]
                and model_name in existing_results[title][ctype]
            ):
                accuracy_dict[title][ctype] = existing_results[title][ctype][model_name]

for title in titles_to_test:
    if title not in contexts:
        print(f"{title} not found")
        continue
    for ctype in context_types:
        if ctype not in contexts[title]:
            print(f"{ctype} not found in {title}")
            continue
        # Skip if already processed for this model
        if (
            title in existing_results
            and ctype in existing_results[title]
            and model_name in existing_results[title][ctype]
        ):
            print(f"Skipping {title} | {ctype} | {model_name}: already processed")
            continue
        context_text = contexts[title][ctype]

        print(f"Processing title: {title} with context type: {ctype}")

        prompts = [
            format_prompt_with_context(context_text, item["question"], item["choices"])
            for item in dataset
        ]
        all_outputs = []
        total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in tqdm(
            range(0, len(prompts), BATCH_SIZE),
            total=total_batches,
            desc="Processing batches",
        ):
            batch_prompts = prompts[i : i + BATCH_SIZE]
            batch_inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
            with torch.inference_mode():
                batch_outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            decoded_outputs = tokenizer.batch_decode(
                batch_outputs, skip_special_tokens=True
            )
            all_outputs.extend([{"generated_text": out} for out in decoded_outputs])

        correct = 0
        total = 0

        for i, item in enumerate(dataset):
            model_output = all_outputs[i]["generated_text"]
            extracted_answer = extract_mc_answer(model_output)
            reference_answer = chr(65 + item["answer"])
            if extracted_answer == reference_answer:
                correct += 1
            total += 1

        accuracy = correct / total
        accuracy_dict[title][ctype] = accuracy
        print(f"{title} | {ctype}: {correct}/{total} correct ({accuracy:.3f})")

        # Write to CSV immediately
        with open(csv_results, "a+", newline="", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            writer = csv.writer(f)
            if f.tell() == 0:  # File is empty, write headers
                headers = [
                    "timestamp",
                    "model_name",
                    "test_name",
                    "context_title",
                    "context_type",
                    "accuracy",
                    "seed",
                ]
                writer.writerow(headers)
            # Write the row
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(
                [
                    timestamp,
                    model_name,
                    "MMLU_selected_subjects",
                    title,
                    ctype,
                    f"{accuracy:.3f}",
                    SEED,
                ]
            )
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

end = time.time()
how_long = end - start
mins, secs = divmod(how_long, 60)
hours, mins = divmod(mins, 60)

print("\n Finished")
print(f"Total runtime: {int(hours)}h {int(mins)}m {secs:.1f}s")
print("\nContext Accuracies:")
print("Overall accuracy by context:", accuracy_dict)
