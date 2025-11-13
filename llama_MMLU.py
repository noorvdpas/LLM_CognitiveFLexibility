# Feedback from Nas: 
# AIM: try to reduce the time taken to run MMLU 
# - Probably would run a lot faster if you construct a dataset or list of questions to be processed in parallel # - Maybe also look at whether you can run the context once into the LLM and then save the state (don't spend too long on this) 
# # AIM: We want to know that improvements are not just random chance 
# - Make sure the outputs are deterministic (i.e. that you get exactly the same response if you re-run the model) 
# - Save the random seed you are using to select questions # 
# AIM: Have a clean structure for loading contexts (and testing them)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import csv
import re
import random
import time
from datetime import datetime
import os

start = time.time()

model_name = "meta-llama/Llama-2-13b-chat-hf"
test_name = "cais/mmlu"

CACHE_DIR = "/scratch/fast/huggingface_cache"
MAX_TOKENS = 30
NUM_TEST = 10
BATCH_SIZE = 2
SEED = 28

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype="auto",
    low_cpu_mem_usage=True,
    cache_dir=CACHE_DIR,
)

tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id


generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device_map=None
)

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
for subject in subjects:
    ds = load_dataset("cais/mmlu", subject, split="test")
    all_data.extend(ds)



#dataset = all_data

random.seed(SEED)
#random.shuffle(all_data)
dataset = random.sample(all_data, NUM_TEST)
#dataset = all_data

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

def format_prompt_with_context(context_text: str, question: str, choices: list[str]) -> str:
    formatted_choices = []
    for i, possible_answer in enumerate(choices):
        answer_letter = chr(65 + i)
        formatted_choice = f"{answer_letter}. {possible_answer}"
        formatted_choices.append(formatted_choice)
    choices_str = "\n".join(formatted_choices)
    return (
        f"{context_text}\n\n"
        f"For the following question, answer ONLY with a single letter (A, B, C, or D):\n\n"
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

contexts_file = "contexts.csv"
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

titles_to_test = ["Astar_(game)"]

#titles_to_test = ["Astar_(game)", "Kingsley_Halt_railway_station", "Epinotia_nemorivaga", "Legal_separation", "Neutral_lipid_storage_disease", "Empty_context"]
context_types = ["clean"]

csv_fullanswers = "mmlu_results_eval.csv"
csv_results = "results.csv"
file_exists = os.path.exists(csv_results)

with open(csv_fullanswers, "w", newline="", encoding="utf-8") as f_full:
    full_writer = csv.DictWriter(
        f_full,
        fieldnames=[
            "subject",
            "context_type",
            "question",
            "model_output",
            "model_answer",
            "reference_answer",
        ],
    )
    full_writer.writeheader()

accuracy_dict = {title: {ctype: 0.0 for ctype in context_types} for title in titles_to_test}

for title in titles_to_test:
    if title not in contexts:
        print(f"{title} not found")
        continue
    for ctype in context_types:
        if ctype not in contexts[title]:
            print(f"{ctype} not found in {title}")
            continue
        context_text = contexts[title][ctype]

        prompts = [format_prompt_with_context(context_text, item["question"], item["choices"]) for item in dataset]
        all_outputs = []
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i+BATCH_SIZE]
            batch_outputs = generator(batch_prompts, max_new_tokens=MAX_TOKENS, do_sample=False)
            for out in batch_outputs:
                if isinstance(out, list):
                    all_outputs.extend(out)
                else:
                    all_outputs.append(out)

        # The batching above is what i had before, which speeds it up a bit but it is still too slow.
        # The problem is the batches are done sequentially. So I treid the method below but now i get GPU out of memory errors. 

        # all_outputs = generator(
        #     prompts,
        #     max_new_tokens=MAX_TOKENS,
        #     batch_size=BATCH_SIZE,
        #     do_sample=False
        # )

        correct = 0
        total = 0

        with open(csv_fullanswers, "a", newline="", encoding="utf-8") as f_full:
            writer = csv.DictWriter(f_full, fieldnames=["subject", "context_title", "context_type", "question", "model_output", "model_answer", "reference_answer"])
            for i, item in enumerate(dataset):
                model_output = all_outputs[i]["generated_text"]
                extracted_answer = extract_mc_answer(model_output)
                reference_answer = chr(65 + item["answer"])
                if extracted_answer == reference_answer:
                    correct += 1
                total += 1
                
                """
                writer.writerow(
                    {
                        "subject": item.get("subject", "unknown"),
                        "context_title": title,
                        "context_type": ctype,
                        "question": f"Q{i+1}: {item['question']}",
                        "model_output": model_output,
                        "model_answer": extracted_answer,
                        "reference_answer": reference_answer,
                    }
                )"""

        accuracy = correct / total
        accuracy_dict[title][ctype] = accuracy
        print(f"{title} | {ctype}: {correct}/{total} correct ({accuracy:.3f})")

with open(csv_results, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        headers = ["timestamp", "model_name", "test_name", "context_title", "context_type", "accuracy", "seed"]
        writer.writerow(headers)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for title, cdict in accuracy_dict.items():
        for ctype, acc in cdict.items():
            writer.writerow([timestamp, model_name, "MMLU_selected_subjects", title, ctype, f"{acc:.3f}", SEED])

end = time.time()
how_long = end - start
mins, secs = divmod(how_long, 60)
hours, mins = divmod(mins, 60)

print("\n Finished")
print(f"Total runtime: {int(hours)}h {int(mins)}m {secs:.1f}s")
print("\nContext Accuracies:")
print("Overall accuracy by context:", accuracy_dict)
