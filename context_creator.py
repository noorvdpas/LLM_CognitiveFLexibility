import spacy
import random
import csv
import os
import string
import spacy
import re
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex


process_text = spacy.load("en_core_web_sm")
text_title = "test"
text = """Pierre Paulus (1881–1959), later Baron Pierre Paulus de Châtelet, was a Belgian expressionist painter. He hasn't been best known as the designer of the "bold rooster" (French: coq hardi) adopted on 3 July 1913 as the symbol of the Walloon Movement and today the flag of Wallonia.[1][2][3] Paulus gained notability during the Walloon Art Exposition of Charleroi in 1911 and, in the interwar period, he held several exhibitions in Europe and in the United States. We've never seen someone like him."""

NEG_CONTRACTIONS = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
}

def expand_neg_contractions(text):
    def replace_match(match):
        word = str(match.group(0))
        lc = word.lower()
        if lc in NEG_CONTRACTIONS:
            new = NEG_CONTRACTIONS[lc]
            if word[0].isupper():
                new = new[0].upper() + new[1:]
                return new
            else:
                return new
        return word
    

    return re.sub(r"\b\w+n't\b", replace_match, text, flags=re.IGNORECASE)



def preprocess(text):
    text = re.sub(r"(\w+)\.((?:\[\d+\])+)", lambda m: m.group(1) + ". " + re.sub(r"(\[\d+\])", r" \1", m.group(2)).strip(),text)
    text = text.replace("’", "'")
    text = expand_neg_contractions(text)
    text = re.sub(r"\s+", " ", text).strip()

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

            elif (token.text == '[' and i + 2 < len(doc) and
                doc[i+1].text.isdigit() and doc[i+2].text == ']'):
                start = i
                end = i + 3
                try:
                    retokenizer.merge(doc[start:end], attrs={"LEMMA": "".join([t.text for t in doc[start:end]])})
                except Exception:
                    pass
                i = end
            else:
                i += 1

            

    return doc


def clean(text):
    doc = preprocess(text)
    return doc



def meaningful_shuffle(text):
    doc = preprocess(text)
    
    citations = set()
    for token in doc:
        if re.fullmatch(r"\[\d+\]", token.text):
            citations.add(token.text)


    pos_list = {}
    sentences = list(doc.sents)

    for sent in sentences:
        for token in sent:
            if token.is_punct:
                continue
            if token.text in citations:
                if "CITATION" not in pos_list:
                    pos_list["CITATION"] = []
                pos_list["CITATION"].append(token.text)
            else:
                label = token.pos_
                if label not in pos_list:
                    pos_list[label] = []
                if label == "PROPN":
                    pos_list[label].append(token.text)
                else:
                    pos_list[label].append(token.text.lower())


    for label in pos_list:
        random.shuffle(pos_list[label])



    final_text = ""
    pos_indices = {label: 0 for label in pos_list}

    for token in doc:
        if token.is_punct or token.text in citations:
            final_text += token.text + token.whitespace_
        else:
            label = token.pos_
            new_word = str(pos_list[label][pos_indices[label]])
            pos_indices[label] += 1
            if token.is_sent_start and token.text[0].isupper():
                new_word = new_word[0].upper() + new_word[1:]
            if token.text.startswith("'") or token.text.startswith("’"):
                if new_word.startswith("'") or new_word.startswith("’"):
                    final_text += new_word + token.whitespace_
                else:
                    final_text += " " + new_word + token.whitespace_
            else: 
                if new_word.startswith("'") or new_word.startswith("’"):
                    final_text = final_text.rstrip(" ")
                    final_text += new_word + token.whitespace_
                else:
                    final_text += new_word + token.whitespace_
            


    return final_text.strip()




def word_shuffle(text):
    doc = preprocess(text)

    citations = set()
    for token in doc:
        if re.fullmatch(r"\[\d+\]", token.text):
            citations.add(token.text)

    words = []
    sentences = list(doc.sents)

    for sent in sentences:
        for token in sent:
            if token.is_punct or token.text in citations:
                continue
            else:
                label = token.pos_
                if label == "PROPN":
                    words.append(token.text)
                else:
                    words.append(token.text.lower())

    random.shuffle(words)

    final_text = ""
    word_idx = 0

    for token in doc:
        if token.is_punct or token.text in citations:
            final_text += token.text + token.whitespace_
        else:
            new_word = str(words[word_idx])
            word_idx += 1
            if token.is_sent_start and token.text[0].isupper():
                new_word = new_word[0].upper() + new_word[1:]
            if token.text.startswith("'") or token.text.startswith("’"):
                if new_word.startswith("'") or new_word.startswith("’"):
                    final_text += new_word + token.whitespace_
                else:
                    final_text += " " + new_word + token.whitespace_
            else: 
                if new_word.startswith("'") or new_word.startswith("’"):
                    final_text = final_text.rstrip(" ")
                    final_text += new_word + token.whitespace_
                else:
                    final_text += new_word + token.whitespace_

    return final_text.strip()
            


def character_shuffle(text):
    doc = preprocess(text)

    citations = set()
    for token in doc:
        if re.fullmatch(r"\[\d+\]", token.text):
            citations.add(token.text)


    chars = []
    for token in doc:
        if not token.is_punct and token.text not in citations:
            for i, c in enumerate(token.text):
                chars.append(c.lower())



    random.shuffle(chars)

    final_text = ""
    for token in doc:
        if token.is_punct or token.text in citations:
            final_text += token.text + token.whitespace_
        else:
            t_length = len(token.text)
            new_word = "".join(chars[:t_length])
            chars = chars[t_length:]
            if token.is_sent_start and token.text[0].isupper():
                new_word = new_word[0].upper() + new_word[1:]
            final_text += new_word + token.whitespace_


    return final_text.strip()



csv_contexts = "contexts_new.csv"
context_file_exists = os.path.isfile(csv_contexts)



with open(csv_contexts, "a", newline="", encoding="utf-8") as w:
    writer = csv.DictWriter(w, fieldnames=["context_title", "context_type", "context_text"])
    if not context_file_exists:
        writer.writeheader()

    writer.writerow(
        {"context_title": text_title, 
        "context_type": "clean", 
        "context_text": preprocess(text)}
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
    
import os
print(os.getcwd())
