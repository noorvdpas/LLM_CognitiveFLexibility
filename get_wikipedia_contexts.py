import wikipedia
import csv
import context_creator


wikipedia.set_lang("en")

num_articles = 100  # Set to 100 for full run

with open("contexts_wikipedia.csv", "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(num_articles):
        success = False
        while not success:
            try:
                random_title = wikipedia.random(1)
                page = wikipedia.page(random_title)
                summary = page.summary.replace("\n", " ").strip()
                if 400 <= len(summary) <= 600:
                    title = page.title.replace(" ", "_")
                    writer.writerow([title, "clean", summary])
                    ms = context_creator.meaningful_shuffle(summary)
                    writer.writerow([title, "meaningful_shuffle", ms])
                    ws = context_creator.word_shuffle(summary)
                    writer.writerow([title, "word_shuffle", ws])
                    cs = context_creator.character_shuffle(summary)
                    writer.writerow([title, "character_shuffle", cs])
                    print(f"Added article {i+1}: {title}")
                    success = True
                else:
                    print(f"Skipped {i+1}: len {len(summary)} not 400-600, retrying...")
            except Exception as e:
                print(f"Error on article {i+1}: {e}, retrying...")

print(f"Finished adding {num_articles} articles.")
