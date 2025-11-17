import pandas as pd
import csv
import context_creator
import random

# URLs for the ConspirED dataset Excel files
training_url = (
    "https://raw.githubusercontent.com/UKPLab/conspired/main/data/context_training.xlsx"
)
testing_url = (
    "https://raw.githubusercontent.com/UKPLab/conspired/main/data/context_testing.xlsx"
)

# Load the datasets
try:
    df_train = pd.read_excel(training_url)
    df_test = pd.read_excel(testing_url)
    df = pd.concat([df_train, df_test], ignore_index=True)
except Exception as e:
    print(f"Error loading datasets: {e}")
    print("Make sure openpyxl is installed: pip install openpyxl")
    exit(1)

# Extract snippets
snippets = df["snippet"].dropna().tolist()

# Set random seed for repeatability
random.seed(42)

# Select a random sample of 100 snippets
if len(snippets) > 100:
    snippets = random.sample(snippets, 100)

num_snippets = len(snippets)

with open("context_negative.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for i, snippet in enumerate(snippets):
        # Clean the snippet: replace newlines with spaces, strip
        snippet = snippet.replace("\n", " ").strip()
        if len(snippet) > 1000:
            snippet = snippet[:1000]
        if snippet:  # Only add if there's content
            title = f"conspiracy_{i+1}"
            writer.writerow([title, "clean", snippet])
            ms = context_creator.meaningful_shuffle(snippet)
            writer.writerow([title, "meaningful_shuffle", ms])
            ws = context_creator.word_shuffle(snippet)
            writer.writerow([title, "word_shuffle", ws])
            cs = context_creator.character_shuffle(snippet)
            writer.writerow([title, "char_shuffle", cs])
            print(f"Added snippet {i+1}: {title}")

print(f"Finished adding {num_snippets} snippets.")
