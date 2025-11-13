import spacy
import random
import csv
import os
import string

SEED = 28
process_text = spacy.load("en_core_web_sm")
text_title = "climate_conspiracy"
text = "And so you’ve got the green movement creating stories that instill fear in the public. You’ve got the media echo chamber - fake news - repeating it over and over and over again to everybody that they’re killing their children. And then you’ve got the green politicians who are buying scientists with government money to produce fear for them in the form of scientific-looking materials. And then you’ve got the green businesses, the rent-seekers, and the crony capitalists who are taking advantage of massive subsidies, huge tax write-offs, and government mandates requiring their technologies to make a fortune on this. And then, of course, you’ve got the scientists who are willingly, they’re basically hooked on government grants."
random.seed(SEED)

def meaningful_shuffle(text):
    doc = process_text(text)

    with doc.retokenize() as retokenizer:
        i = 0
        while i < len(doc):
            token = doc[i]
            if token.text.isalpha() and i + 2 < len(doc) and doc[i + 1].text == '-' and doc[i + 2].text.isalpha():
                start = i
                end = i + 3
                while end + 1 < len(doc) and doc[end].text == '-' and doc[end + 1].text.isalpha():
                    end += 2
                try:
                    retokenizer.merge(doc[start:end])
                except Exception:
                    pass
                i = end
            else:
                i += 1

    pos_list = {}
    sentences = list(doc.sents)

    for sent in sentences:
        for token in sent:
            if token.is_punct:
                continue
            label = token.pos_
            if label not in pos_list:
                pos_list[label] = []
            if label == "PROPN":
                pos_list[label].append(token.text)
            else:
                pos_list[label].append(token.text.lower())

    for label in pos_list:
        random.shuffle(pos_list[label])

    new_text = []
    pos_indices = {label: 0 for label in pos_list}

    for token in doc:
        if token.is_punct:
            new_text.append(token.text)
        else:
            label = token.pos_
            new_word = pos_list[label][pos_indices[label]]
            pos_indices[label] += 1
            if token.is_sent_start and token.text[0].isupper():
                new_word = new_word.capitalize()
            new_text.append(new_word)

    final_text = ""
    for token in new_text:
        if token in string.punctuation and token not in ["(", "[", "{", "'"]:
            final_text = final_text.rstrip() + token
        elif token in ["(", "[", "{"]:
            final_text += " " + token
        elif len(final_text) > 0 and final_text[-1] in ["(", "[", "{", "'", "/"]:
            final_text += token
        elif token in ["'s", "'d", "'re", "'m", "'ve"]:
            final_text = final_text.rstrip() + token
        else:
            final_text += " " + token

    final_text = final_text.strip()
    return final_text



def word_shuffle(text):
    doc = process_text(text)

    with doc.retokenize() as retokenizer:
        i = 0
        while i < len(doc):
            token = doc[i]
            if token.text.isalpha() and i + 2 < len(doc) and doc[i + 1].text == '-' and doc[i + 2].text.isalpha():
                start = i
                end = i + 3
                while end + 1 < len(doc) and doc[end].text == '-' and doc[end + 1].text.isalpha():
                    end += 2
                try:
                    retokenizer.merge(doc[start:end])
                except Exception:
                    pass
                i = end
            else:
                i += 1

    sentences = list(doc.sents)
    words = []

    for sent in sentences:
        for token in sent:
            if token.is_punct:
                continue
            elif token.pos_ == "PROPN":
                words.append(token.text)
            else:
                words.append(token.text.lower())

    random.shuffle(words)

    new_text = []
    word_idx = 0

    for token in doc:
        if token.is_punct:
            new_text.append(token.text)
        else:
            new_word = words[word_idx]
            word_idx += 1
            if token.is_sent_start and token.text[0].isupper():
                new_word = new_word.capitalize()
            new_text.append(new_word)

    final_text = ""
    for token in new_text:
        if token in string.punctuation and token not in ["(", "[", "{", "'"]:
            final_text = final_text.rstrip() + token
        elif token in ["(", "[", "{"]:
            final_text += " " + token
        elif len(final_text) > 0 and final_text[-1] in ["(", "[", "{", "'", "/"]:
            final_text += token
        elif token in ["'s", "'d", "'re", "'m", "'ve"]:
            final_text = final_text.rstrip() + token
        else:
            final_text += " " + token

    final_text = final_text.strip()
    return final_text

def character_shuffle(text):
    doc = process_text(text)

    with doc.retokenize() as retokenizer:
        i = 0
        while i < len(doc):
            token = doc[i]
            if token.text.isalpha() and i + 2 < len(doc) and doc[i + 1].text == '-' and doc[i + 2].text.isalpha():
                start = i
                end = i + 3
                while end + 1 < len(doc) and doc[end].text == '-' and doc[end + 1].text.isalpha():
                    end += 2
                try:
                    retokenizer.merge(doc[start:end])
                except Exception:
                    pass
                i = end
            else:
                i += 1

    chars = []
    for c in text:
        if c == '-':
            chars.append(c)
        elif c not in string.punctuation and not c.isspace():
            chars.append(c.lower())

    random.shuffle(chars)
    new_text = []

    for token in doc:
        if token.is_punct:
            new_text.append(token.text)
        else:
            t_length = len(token.text)
            new_word = "".join(chars[:t_length])
            chars = chars[t_length:]
            if token.text[0].isupper():
                new_word = new_word.capitalize()
            new_text.append(new_word)

    final_text = ""
    for token in new_text:
        if token in string.punctuation and token not in ["(", "[", "{", "'"]:
            final_text = final_text.rstrip() + token
        elif token in ["(", "[", "{"]:
            final_text += " " + token
        elif len(final_text) > 0 and final_text[-1] in ["(", "[", "{", "'", "/"]:
            final_text += token
        elif token in ["'s", "'d", "'re", "'m", "'ve"]:
            final_text = final_text.rstrip() + token
        else:
            final_text += " " + token

    final_text = final_text.strip()
    return final_text

csv_contexts = "contexts_bad.csv"
context_file_exists = os.path.isfile(csv_contexts)

with open(csv_contexts, "a", newline="", encoding="utf-8") as w:
    writer = csv.DictWriter(w, fieldnames=["context_title", "context_type", "context_text"])
    if not context_file_exists:
        writer.writeheader()

    writer.writerow(
        {"context_title": text_title, 
        "context_type": "clean", 
        "context_text": text}
        )
    writer.writerow(
        {"context_title": text_title, 
        "context_type": "meaningful_shuffle", 
        "context_text": meaningful_shuffle(text)}
        )
    writer.writerow(
        {"context_title": text_title, 
        "context_type": "word_shuffle", 
        "context_text": word_shuffle(text)}
        )
    writer.writerow(
        {"context_title": text_title, 
        "context_type": "char_shuffle", 
        "context_text": character_shuffle(text)}
        )

