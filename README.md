# rag-document-advisor — Cloud RAG for Legal Documents

> Upload any PDF. Ask questions. Get answers with exact source pages — in seconds.

`rag-document-advisor` is a Streamlit-based RAG application for legal and structured documents. It handles scanned PDFs via OCR, indexes content with hybrid search (vector + BM25), and returns grounded answers with page-level citations using Groq's LLM API.

Built as the cloud-first variant in a [3-part RAG series](https://github.com/Muneshshaganti) — see also [`Rag_Ollama`](https://github.com/Muneshshaganti/Rag_Ollama) (offline) and [`multirag`](https://github.com/Muneshshaganti/multirag) (multi-model fallback).

---

## What problem this solves

Legal documents are long, dense, and often scanned. Reading a 60-page contract to answer one question takes 30+ minutes. This system reduces that to under 30 seconds — with the exact page number so you can verify the source yourself.

---

## Features

- **Scanned PDF support** — Tesseract OCR extracts text from image-based pages
- **Hybrid retrieval** — vector search (Chroma + HuggingFace embeddings) + BM25 keyword search
- **Source attribution** — every answer includes the page number(s) it was drawn from
- **Context-aware answers** — Groq LLM answers only from retrieved document context, not general knowledge
- **Simple UI** — Streamlit interface, no configuration required

---

## Architecture

```
PDF → OCR (if scanned) → Chunking → Chroma (vectors) + BM25 (keywords)
                                           │
                              User question → Hybrid retrieval
                                           │
                                    Groq LLM (Llama 3)
                                           │
                              Answer + Source pages → UI
```

---

## Tech stack

| Component | Technology |
|---|---|
| LLM | Groq — LLaMA 3 |
| Embeddings | HuggingFace sentence-transformers |
| Vector store | Chroma |
| Keyword search | BM25 |
| OCR | Tesseract + pdf2image |
| UI | Streamlit |
| Dev container | `.devcontainer` (VS Code / Codespaces ready) |

---

## Quickstart

```bash
git clone https://github.com/Muneshshaganti/rag-document-advisor.git
cd rag-document-advisor
pip install -r requirements.txt
```

Set your API key:

```bash
export GROQ_API_KEY=your_key_here
```

Run:

```bash
streamlit run main.py
```

---

## Related projects

| Repo | Key difference |
|---|---|
| This repo | Cloud LLM (Groq), simple setup |
| [`Rag_Ollama`](https://github.com/Muneshshaganti/Rag_Ollama) | Local LLM (Ollama), zero API cost, DB persistence |
| [`multirag`](https://github.com/Muneshshaganti/multirag) | Multi-model fallback, hybrid retrieval, multilingual |

---

## Topics

`rag` `retrieval-augmented-generation` `langchain` `groq` `chroma` `ocr` `streamlit` `document-ai` `legal-tech` `python` `huggingface` `bm25`
# RAG Document Advisor

This project is an AI-powered document assistant that allows users to upload a PDF and ask questions about its content.

The system extracts text using OCR, stores document chunks in a vector database, and retrieves relevant context to generate answers using a Large Language Model.

## Features
- Upload scanned PDFs
- OCR text extraction
- Hybrid search (Vector + BM25)
- Context-aware answers
- Source page references

## Tech Stack
- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- Chroma Vector Database
- Groq LLM
- Tesseract OCR


