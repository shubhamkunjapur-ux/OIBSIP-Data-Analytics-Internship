<div align="center">

# ⌨️ Autocomplete & Autocorrect Using NLP

**Oasis Infobyte (OIBSIP) · Data Analytics Internship · Level 2 – Task 5**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-2E7D32?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

[🚀 Live Dashboard](https://oibsip-autocomplete-autocorrect-nlpgit-fqebdpk5l7obck7jyqrxts.streamlit.app/) · [🎥 Demo Video](https://drive.google.com/file/d/1bEOyAglaVUjX8ukrkgM-UkmOSGAdnW4O/view?usp=sharing) · [📚 Dataset](https://huggingface.co/datasets/Salesforce/wikitext)

</div>

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Evaluation Highlights](#-evaluation-highlights)
- [Project Links](#-project-links)
- [Tech Stack](#-tech-stack)
- [Run Locally](#-run-locally)
- [Limitations & Future Scope](#-limitations--future-scope)

## 📌 Project Overview
This project implements and evaluates a text autocomplete and autocorrect system using the **WikiText-2** corpus. It combines frequency-based N-gram language models for next-word suggestions with edit-distance-based spelling correction, and presents the results in an interactive Streamlit dashboard.

## ✨ Key Features
- Text preprocessing, tokenization, and word-frequency analysis
- **Bigram and Trigram** next-word prediction with fallback suggestions
- **Levenshtein Edit Distance** and vocabulary-frequency-based autocorrect
- Model comparison using **Precision@3** and **Recall@3**
- Spelling-correction evaluation, charts, and confusion matrix
- Interactive Streamlit deployment

## 📊 Evaluation Highlights

| Evaluation | Result |
|---|---:|
| Bigram Recall@3 | 25.23% |
| Trigram with fallback Recall@3 | **27.48%** |
| Exact correction accuracy on 60 synthetic typos | **91.67%** |

> **Note:** Results are from the project notebook's held-out next-word evaluation and synthetic typo tests. They should not be interpreted as production-level performance.

## 🔗 Project Links

| Resource | Link |
|---|---|
| 🚀 **Live Streamlit Dashboard** | [Open NLP Dashboard](https://oibsip-autocomplete-autocorrect-nlpgit-fqebdpk5l7obck7jyqrxts.streamlit.app/) |
| 🎥 **Demo Video** | [Watch Project Walkthrough](https://drive.google.com/file/d/1bEOyAglaVUjX8ukrkgM-UkmOSGAdnW4O/view?usp=sharing) |
| 📚 **Dataset** | [WikiText-2 on Hugging Face](https://huggingface.co/datasets/Salesforce/wikitext) |

## 🛠️ Tech Stack
`Python` · `NLP` · `NLTK` · `Pandas` · `Scikit-learn` · `Plotly` · `Streamlit`

## ▶️ Run Locally
Keep these files in the same folder (the confusion-matrix PNG is optional, only needed to display that image):

```text
├── app.py
├── requirements.txt
├── model_artifacts.pkl.gz
├── summary_metrics.json
└── top20_words.csv
```

Then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Limitations & Future Scope
N-gram models have limited context, and one-edit autocorrect can miss complex or context-dependent errors.

**Future extensions:**
- Transformer-based prediction
- Context-aware correction
- Personalized suggestions
- Evaluation using real-world typing mistakes

---

<div align="center">

*Developed as part of the Oasis Infobyte Data Analytics Internship.*

</div>
