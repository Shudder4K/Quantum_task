import pandas as pd
import spacy
import requests
import random
from bs4 import BeautifulSoup
from spacy import displacy
from spacy.tokens import DocBin
nlp = spacy.load("en_core_web_sm")


# Function for creation of NER-tags
def generate_conll_format(texts, entities):
    dataset = []
    for text in texts:
        tokens = text.split()  # split into words
        labels = ["O"] * len(tokens)

        # Позначаємо назви гір у тексті
        for entity in entities:
            for i, token in enumerate(tokens):
                if token.startswith(entity):
                    labels[i] = "B-MOUNTAIN" if labels[i] == "O" else "I-MOUNTAIN"

        # adding tokens and tags in dataset
        dataset.extend(zip(tokens, labels))
        dataset.append(("", ""))  # Empty line for dividing sentences

    return dataset





mountains = ["Everest", "Kilimanjaro", "Mont-Blanc", "Aconcagua", "Lhotse", "Nuptse"]

# Text generation with mountains names
texts = [
     f"Mount {mountains[0]}, standing at an impressive 8,848 meters, {mountains[0]} is the highest mountain on Earth and a beacon for adventurers worldwide. Located in the majestic Himalayas, {mountains[0]} is flanked by other towering peaks like {mountains[4]} and {mountains[5]}. In contrast, Mount {mountains[1]} in Tanzania is the tallest freestanding mountain, rising 5,895 meters above the plains. {mountains[1]} snowy summit stands as a striking contrast to the surrounding African savanna. The Alps, stretching across Europe, are home to {mountains[2]}, which reaches a height of 4,807 meters. {mountains[2]} have inspired climbers and artists alike, offering breathtaking views and challenging trails. Meanwhile, the Andes, the world's longest mountain range, run down the spine of South America, boasting the mighty {mountains[3]} at 6,961 meters, the tallest mountain outside Asia. Each mountain tells a unique story, shaped by geological forces over millions of years. They host diverse ecosystems, from alpine meadows to glacial ice caps, and are home to unique flora and fauna adapted to extreme conditions. Mountains like the Rockies, the Pyrenees, and the Carpathians not only define landscapes but also the cultures and histories of the people who live in their shadows."
    for mountain in mountains
]
dataset = generate_conll_format(texts, mountains)

# Saving in file
with open("ner_dataset.conll", "w") as file:
    for token, label in dataset:
        file.write(f"{token} {label}\n")


