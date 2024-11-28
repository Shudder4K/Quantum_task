### **README for Notebook 1: Named Entity Recognition for Mountain Names**

---

# **Named Entity Recognition for Mountain Names**

This notebook demonstrates the development of a Named Entity Recognition (NER) pipeline for identifying and tagging mountain names in a text corpus. The process involves generating a custom NER dataset, training a BERT-based model, and visualizing the results.

---

## **Notebook Overview**

### **1. Generating the NER Dataset**
- **Purpose**: Create a labeled dataset for training an NER model on mountain-related texts.
- **Process**:
  1. Define a list of mountain names (e.g., "Everest", "Kilimanjaro").
  2. Generate text samples embedding these names in context.
  3. Annotate the texts with `B-MOUNTAIN` (beginning of a mountain name) and `I-MOUNTAIN` (inside a mountain name) tags.
  4. Save the dataset in CoNLL format.

---

### **2. Parsing the Dataset**
- **Objective**: Read the CoNLL file and prepare it for model training.
- **Steps**:
  1. Parse sentences and their corresponding labels from the CoNLL file.
  2. Structure the data for tokenization and alignment.

---

### **3. Training the NER Model**
- **Purpose**: Fine-tune a pre-trained BERT model for mountain name recognition.
- **Process**:
  1. Use the `BertTokenizerFast` to tokenize sentences while aligning labels.
  2. Fine-tune `BertForTokenClassification` using the tokenized dataset.
  3. Split the dataset into training and validation sets for evaluation.
  4. Configure and run the `Trainer` for model training.

---

### **4. Inference and Visualization**
- **Objective**: Apply the trained NER model to identify mountain names in new text.
- **Steps**:
  1. Use the trained model to predict entities in sample text.
  2. Convert predictions into a Spacy-compatible format.
  3. Render visualizations using Spacy's `displacy` and save them as HTML.

---

## **Outputs**
- Trained NER model saved to disk.
- Annotated text with identified mountain names.
- Visualization of NER results in HTML format.

---

### **Requirements**
- Libraries: `transformers`, `torch`, `spacy`, `sklearn`, `bs4`
- Dataset: Custom-generated CoNLL file with mountain names and their contexts.

---

### **Conclusion**
This notebook provides a step-by-step guide for training a custom NER model to identify mountain names in text. The method can be adapted for other domains requiring domain-specific NER models.

---
