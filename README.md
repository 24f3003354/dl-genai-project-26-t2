# Smart MCQ Solver

An AI/ML-based system for solving and ranking answers to multiple-choice questions using traditional machine learning, semantic retrieval, RAG, and transformer-based models.

> Built as part of the **Deep Learning and Generative AI** coursework at IIT Madras, exploring and comparing different approaches for MCQ answer prediction.

## 🚀 Demo

**Live Demo:** [Try Smart MCQ Solver](https://huggingface.co/spaces/DeveloperFardeen/SmartMCQSolver)

**Project Report:** [Read the Project Report](https://drive.google.com/file/d/1l-D8mkgjQFl0vHQrAcN1_2W6MgR-QqKK/view?usp=drive_link)

---

## Overview

Smart MCQ Solver explores multiple approaches for predicting the correct answer to multiple-choice questions.

The project compares:

- TF-IDF + MLP
- Zero-shot semantic similarity
- Retrieval-Augmented Generation (RAG)
- Transformer-based classification
- LoRA fine-tuning

The goal was not just to build a solver, but to experiment with different NLP and information-retrieval techniques and evaluate their performance systematically.

---

## 🧠 Approach

The project experimented with three major pipelines.

### 1. TF-IDF + MLP

A traditional machine-learning baseline using:

- TF-IDF vectorization
- Multi-Layer Perceptron
- Question and option text features

This approach achieved the strongest final validation/Kaggle performance among the evaluated approaches.

### 2. Semantic Retrieval + Reranking

A retrieval pipeline using:

- Sentence Transformers / MiniLM
- FAISS vector index
- Semantic similarity search
- Cross-Encoder reranking

Questions and candidate answers are converted into embeddings and retrieved based on semantic similarity before being reranked by a Cross-Encoder.

### 3. RAG + Transformer Fine-tuning

An experimental RAG pipeline combining:

- FAISS retrieval
- MiniLM embeddings
- DeBERTa
- Cross-Encoder reranking
- LoRA fine-tuning

The objective was to investigate whether retrieved contextual information could improve transformer-based answer prediction.

---

## 📊 Results

The project achieved a final **MAP@3 score of 0.752** with the TF-IDF + MLP approach.

| Approach | Key Techniques |
|---|---|
| TF-IDF + MLP | TF-IDF, MLP |
| Semantic Retrieval | MiniLM, FAISS |
| RAG | MiniLM, FAISS, RAG |
| Reranking | Cross-Encoder |
| Fine-tuning | DeBERTa, LoRA |

The experiments showed that a relatively simple feature-based model could outperform the more complex retrieval and transformer pipelines on this dataset.

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- PyTorch
- Scikit-learn

### NLP
- Sentence Transformers
- MiniLM
- DeBERTa
- TF-IDF
- Cross-Encoder
- Semantic Similarity

### Retrieval
- FAISS
- Vector Embeddings
- RAG

### Fine-tuning
- LoRA

---

## 🔄 Pipeline

```text
                 ┌─────────────────┐
                 │   MCQ Dataset   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Text Processing │
                 └────────┬────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       ┌──────────────┐       ┌────────────────┐
       │ TF-IDF + MLP │       │   Embeddings   │
       └──────┬───────┘       └───────┬────────┘
              │                       │
              │                       ▼
              │                ┌──────────────┐
              │                │ FAISS Search │
              │                └──────┬───────┘
              │                       │
              │                       ▼
              │                ┌──────────────┐
              │                │ Cross-Encoder│
              │                │  Reranking   │
              │                └──────┬───────┘
              │                       │
              │                       ▼
              │                  ┌─────────┐
              │                  │   RAG   │
              │                  └────┬────┘
              │                       │
              └───────────┬───────────┘
                          ▼
                  ┌───────────────┐
                  │ Answer Ranking│
                  └───────────────┘
```
