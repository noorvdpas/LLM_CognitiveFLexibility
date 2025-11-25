import pandas as pd
import csv
import context_creator
import random

# Load the ISOT fake news datasets
try:
    df_fake = pd.read_csv("datasets/ISOT/Fake.csv")
    df_true = pd.read_csv("datasets/ISOT/True.csv")
    # Add labels
    df_fake["label"] = "fake"
    df_true["label"] = "real"
except Exception as e:
    print(f"Error loading datasets: {e}")
    print("Make sure the CSV files are in the current directory")
    exit(1)


# Function to process a dataset
def process_dataset(df, label, output_file):
    snippets = df["text"].dropna().tolist()

    # Set random seed for repeatability
    random.seed(42)

    # Shuffle snippets for random selection
    random.shuffle(snippets)

    num_snippets = 0
    max_snippets = 500

    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for snippet in snippets:
            if num_snippets >= max_snippets:
                break
            # Clean the snippet: replace newlines with spaces, strip
            snippet = snippet.replace("\n", " ").strip()
            if 400 <= len(snippet) <= 600:
                title = f"isot_{label}_{num_snippets+1}"
                writer.writerow([title, "clean", snippet])
                ms = context_creator.meaningful_shuffle(snippet)
                writer.writerow([title, "meaningful_shuffle", ms])
                ws = context_creator.word_shuffle(snippet)
                writer.writerow([title, "word_shuffle", ws])
                cs = context_creator.character_shuffle(snippet)
                writer.writerow([title, "character_shuffle", cs])
                print(f"Added {label} snippet {num_snippets+1}: {title}")
                num_snippets += 1

    print(f"Finished adding {num_snippets} {label} snippets to {output_file}.")


# Process fake news
process_dataset(df_fake, "fake", "contexts_isot_fake.csv")

# Process true news
process_dataset(df_true, "true", "contexts_isot_true.csv")
