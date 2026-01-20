# AI-Based Health Claim Verification using DeBERTa v3 and Retrieval-Augmented Generation

This repository implements an AI system for verifying health-related claims
using a fine-tuned **DeBERTa v3** model combined with **Retrieval-Augmented
Generation (RAG)** for evidence grounding and explanation.

The system is designed to combat medical misinformation by producing
both **claim validity predictions** and **human-readable explanations**
supported by retrieved evidence.

---

## Problem Statement
Health-related misinformation poses significant risks to public safety.
Manual fact-checking is slow and does not scale, while black-box models
lack transparency.

This project addresses these challenges by:
- Verifying health claims using a supervised transformer model
- Grounding predictions in retrieved evidence
- Generating explanations to improve trust and interpretability

---

## System Overview
The pipeline consists of three main components:

1. **Claim Verification Model**
   - Fine-tuned DeBERTa v3 transformer
   - Classifies health claims as supported or unsupported

2. **Retrieval Module**
   - FAISS-based dense vector index
   - Retrieves relevant documents for a given claim

3. **Explanation Generation**
   - Combines model predictions with retrieved evidence
   - Produces interpretable natural language explanations

---

## Model Details
- Base model: DeBERTa v3
- Fine-tuning: Supervised classification on health claim datasets
- Model format: HuggingFace-compatible (`safetensors`)

---

## Data
- Training and evaluation data stored in TSV and CSV formats
- Includes labeled health claims and supporting evidence
- Preprocessing applied to normalize and clean input text

---

## Retrieval
- FAISS index built over health-related textual evidence
- Enables fast similarity-based retrieval during inference
- Supports retrieval-augmented reasoning

---


