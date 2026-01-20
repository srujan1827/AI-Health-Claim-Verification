import os
import textwrap
import pandas as pd
import numpy as np
import faiss
import requests
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from openai import OpenAI

DATA_DIR = "data"
MODEL_DIR = "model"
FAISS_PATH = "retrieval/faiss_index_final.bin"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def web_retrieve_pubmed(query, k=3):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    try:
        search = requests.get(
            f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={k}&retmode=json",
            timeout=10
        )
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        ids_str = ",".join(ids)
        summary = requests.get(
            f"{base_url}esummary.fcgi?db=pubmed&id={ids_str}&retmode=json",
            timeout=10
        )
        result = summary.json().get("result", {})

        evidence = []
        for pid in ids:
            item = result.get(pid, {})
            title = item.get("title", "")
            source = item.get("source", "")
            pubdate = item.get("pubdate", "")
            if title:
                evidence.append(f"{title} ({source}, {pubdate})")

        return evidence

    except Exception as e:
        print("PubMed retrieval failed:", e)
        return []


def rag_fact_check(claim, explanation, k=3):
    corpus_df = pd.read_csv(f"{DATA_DIR}/Datensatz_final.csv")
    corpus_df["text"] = (
        corpus_df["claim"].fillna("") + " " +
        corpus_df["explanation"].fillna("")
    )

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.read_index(FAISS_PATH)

    query = f"{claim} {explanation}"
    query_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")

    _, I = index.search(query_emb, k)
    retrieved = [corpus_df.iloc[i]["text"] for i in I[0]]

    if not retrieved:
        retrieved = web_retrieve_pubmed(claim, k=k)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )

    result = classifier(query)[0]
    label_map = {
        "LABEL_0": "False",
        "LABEL_1": "Partly True",
        "LABEL_2": "True"
    }

    label = label_map.get(result["label"], result["label"])
    confidence = round(result["score"] * 100, 2)

    prompt = f"""
    You are a medical fact-checking assistant.

    Claim: {claim}
    Explanation: {explanation}

    Evidence:
    {retrieved}

    Model Verdict: {label} ({confidence}% confidence)

    Write a concise reasoning grounded in evidence.
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=180,
    )

    print("-" * 80)
    print("Verdict:", label, f"({confidence}%)")
    print("Reasoning:")
    print(textwrap.fill(response.output_text.strip(), width=100))
    print("-" * 80)


if __name__ == "__main__":
    rag_fact_check(
        claim="Example medical claim",
        explanation="Example explanation"
    )
