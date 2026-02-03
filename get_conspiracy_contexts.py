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
random.seed(28)

# Shuffle all snippets for random selection
random.shuffle(snippets)

num_snippets = 0
max_snippets = 1000

# Set random seed for repeatability
random.seed(28)

textt = "The Al Jazeera hoax was intended to create the impression that Tripoli had fallen so as:   (1) to break the Libyan resistance by creating panic and chaos in the Libyan captial.  (2) to provide cover for the massacres of civilians that would occur in the days following the declaration of rebel victory.   In other words, the media would provide cover for the war crimes and crimes against humanity that are necessary in order to subjugate the Libyan Jamhahirya to Western corporate interests."


with open("contexts_conspiracy.csv", "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for snippet in snippets:
        if num_snippets >= max_snippets:
            break
        # Clean the snippet: replace newlines with spaces, strip
        snippet = snippet.replace("\n", " ").strip()
        if 200 <= len(snippet) <= 800:
            title = f"conspiracy_{num_snippets+1}"
            cl = context_creator.clean(snippet)
            writer.writerow([title, "clean", cl])
            ms = context_creator.meaningful_shuffle(snippet)
            writer.writerow([title, "meani", ms])
            ws = context_creator.word_shuffle(snippet)
            writer.writerow([title, "wordd", ws])
            cs = context_creator.character_shuffle(snippet)
            writer.writerow([title, "chara", cs])
            print(f"Added snippet {num_snippets+1}: {title}")
            num_snippets += 1

print(f"Finished adding {num_snippets} snippets.")
