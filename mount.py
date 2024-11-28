import torch
import spacy
from spacy import displacy
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification, Trainer, TrainingArguments
from transformers import pipeline
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def parse_conll(file_path):
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                token, tag = line.split()
                current_sentence.append(token)
                current_labels.append(tag)

    # Add the last sentence if file does not end with a blank line
    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)

    return sentences, labels

def tokenize_and_align_labels(sentences, labels):
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to original words
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore [CLS], [SEP], and padding
            elif word_idx != previous_word_idx:
                label_ids.append(tag2id[label[word_idx]])
            else:
                label_ids.append(tag2id[label[word_idx]])  # Align with subword
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

def convert_to_spacy_format(text, results):
    entities = []
    for entity in results:
        start = entity["start"]
        end = entity["end"]
        label = entity["entity_group"]
        entities.append({"start": start, "end": end, "label": label})
    return {"text": text, "ents": entities, "title": "Named Entities"}

class NERDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.data.items()}


# Load the CoNLL file
file_path = "ner_dataset.conll"
sentences, labels = parse_conll(file_path)

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

# Map tags to indices
tag2id = {"O": 0, "B-MOUNTAIN": 1}
id2tag = {v: k for k, v in tag2id.items()}


# Tokenize and align
tokenized_data = tokenize_and_align_labels(sentences, labels)
print(tokenized_data)

# Load pre-trained BERT model
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(tag2id))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=50,
    weight_decay=0.01,
    logging_dir="./logs",
)

train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)

# Tokenize validation data
val_tokenized_data = tokenize_and_align_labels(val_sentences, val_labels)

# Define datasets
train_dataset = NERDataset(tokenized_data)
eval_dataset = NERDataset(val_tokenized_data)
train_dataset = NERDataset(tokenized_data)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

trainer.save_model("./bert-ner")

# Load fine-tuned model
ner_model = pipeline("ner", model="./bert-ner", tokenizer="./bert-ner", aggregation_strategy="simple")

# Input text
text = "Mount Everest is the biggest mountain in the world, Kilimonjaro is the biggest mountain in Africa"

inputs = tokenizer("Mount Everest is the biggest mountain", return_offsets_mapping=True, truncation=True)
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
print("Offsets:", inputs["offset_mapping"])  # Check how tokens align with text

# Perform inference
results = ner_model(text)
print(results)

spacy_data = convert_to_spacy_format(text, results)

# Рендеринг у HTML
html = displacy.render(spacy_data, style="ent", manual=True)

# Збереження в HTML-файл
output_file = "ner_visualization.html"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(html)

print(f"NER visualization saved to {output_file}")