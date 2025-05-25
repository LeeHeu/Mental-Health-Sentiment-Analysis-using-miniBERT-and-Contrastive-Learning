
# 🧠 Mental Health Prediction using MiniBERT + Contrastive Learning

## 🔍 Overview

This project aims to predict users' **mental health status** based on short written statements by leveraging **MiniBERT embeddings** and **contrastive learning**.  
It combines powerful pre-trained transformer models with representation learning to better distinguish between similar yet semantically distinct text samples.

---

## 🗃️ Dataset

- **File:** `Combined Data (1).csv`
- **Size:** 53,043 records
- **Columns:**
  - `statement`: textual input from users
  - `status`: mental health condition (e.g., *healthy*, *depressed*)
- Preprocessing includes:
  - Cleaning, tokenization
  - BERT embeddings extraction
  - Label encoding for supervised learning

---

## 🧪 Methodology

### 🧠 Model Pipeline

- **Text Encoder:** `MiniBERT` from `transformers` (lightweight BERT variant)
- **Contrastive Learning Module:**
  - Generates positive and negative sample pairs
  - Learns better representations via triplet loss / cosine similarity
- **Classifier Head:**
  - Fully connected layers for final prediction

### ⚙️ Techniques Used

- HuggingFace Transformers (`MiniBERT`)
- Sentence Embedding Extraction
- Siamese Network / Contrastive Loss
- Evaluation Metrics:
  - Accuracy, F1-score, Confusion Matrix
  - Visualization with t-SNE or PCA (optional)

---

## 📂 Project Structure

```
├── nlp-minbert-for-mental_health.ipynb   # Main notebook
├── Combined Data (1).csv                 # Mental health dataset
├── README.md                             # Project documentation
```

---

## 📈 Sample Results

- MiniBERT with contrastive learning yields improved separation of similar textual samples (e.g., anxious vs. depressed).
- Classifier achieved high F1-score on imbalanced label sets.
- The model can be further fine-tuned or used in clinical decision support tools.

---

## ⚙️ Requirements

```bash
Python >= 3.8

# Required libraries
pip install transformers torch pandas scikit-learn matplotlib seaborn
```

---

## 🚀 To Run the Notebook

```python
# Open nlp-minbert-for-mental_health.ipynb
- Execute all cells from top to bottom
- Ensure internet access to download HuggingFace MiniBERT model
```

---

## 👨‍⚕️ Ethical Consideration

This model is a **research prototype** and should **not be used for clinical diagnosis**. Always consult mental health professionals for real-world use.

---

## 👩‍💻 Author

Project by: **VO LE HIEU**  
Course: NLP / Deep Learning for Mental Health

---
