Track B

---
title: RAG Architect Advanced
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.49.1
app_file: app.py
pinned: false
---

# 🧬 Advanced RAG Architect

A high-precision Retrieval-Augmented Generation (RAG) system utilizing a hybrid pipeline of **Pinecone** (Vector Store), **Cohere** (Embeddings & Reranking), and **Google Gemini** (Generation) to deliver accurate, citation-backed answers from user-uploaded PDFs or raw text.

## 👤 Author
*   **Name:** Sridhar K
*   **Resume:** https://drive.google.com/file/d/1ci1jmKH14PK7IzuIb5Ei48zBZPNT1b3W/view

---

## 🏗️ Architecture

The system implements a **multi-stage retrieval pipeline** (Track B) designed to maximize context relevance and minimize hallucinations.

### 1. **Ingestion Layer**
*   **Source:** PDF Documents or Raw Text.
*   **Chunking:** `RecursiveCharacterTextSplitter` (Size: 1000, Overlap: 150).
*   **Embedding Model:** `cohere.embed-english-v3.0` (Dimension: 1024).
*   **Storage:** Pinecone Serverless Index.

### 2. **Retrieval Layer**
*   **Vector Search:** Top-20 semantic search via Pinecone.
*   **Neural Reranking:** Top-5 reranking via `cohere.rerank-english-v3.0`.
*   **Logic:** Simple vector search captures broad context, while the reranker filters out irrelevant chunks that happen to share keywords, ensuring high precision.

### 3. **Generation Layer**
*   **LLM:** `gemini-2.0-flash` (Google DeepMind).
*   **Prompting:** Strict instruction set requiring inline citations (e.g., `[1]`) based *only* on provided context.

---

## ⚙️ Index Configuration (Track B)

**Vector Database:** Pinecone
*   **Index Name:** `rag-app` (Configurable via .env)
*   **Dimension:** 1024 (Matches Cohere v3 embeddings)
*   **Metric:** `cosine` (Optimized for semantic similarity)
*   **Pod Type:** Serverless (AWS/Starter)

---

## 🚀 Setup & Installation

### Requirements
*   Python 3.10+
*   API Keys: Pinecone, Cohere, Google Gemini

### 1. Clone & Install
```bash
git clone https://github.com/srihar-15/RAG-Architecture
cd rag-architect
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory:
```ini
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=rag-app
COHERE_API_KEY=your_cohere_key
GEMINI_API_KEY=your_gemini_key
```

### 3. Run Locally
```bash
streamlit run app.py
```

---

## 📝 Remarks

### **Limits & Trade-offs**
1.  **Latency vs. Precision:** Reranking adds ~200-500ms to the query time but drastically improves answer quality. We prioritized precision over raw speed.
2.  **Statelessness:** The current implementation clears the vector index on every new ingestion. It is designed for single-session analysis rather than a persistent multi-user knowledge base.
3.  **PDF Parsing:** Uses basic `pypdf`. Complex PDFs with tables or multi-column layouts may have degraded extraction quality compared to OCR-based solutions (e.g., Unstructured.io).
4.  **Cohere Chunk Limit:** This system allows a maximum of 500 chunks per document to respect API rate limits and processing constraints.
5.  **Single Concurrency:** The system is designed for a single active user. Since it performs a `delete_all` on the vector index for every new upload, simultaneous users would overwrite each other's data.
6.  **Language Support:** The embedding model (`embed-english-v3.0`) and reranker are optimized specifically for English text. Performance on other languages may vary.

### **Future Improvements**
*   **Hybrid Search:** Implement sparse-dense vectors (Splade + Dense) to better catch keyword-specific queries.
*   **Multi-Modal support:** Upgrade Gemini integration to parse charts and images from PDFs.
*   **Evaluation Pipeline:** Integrate **Ragas** to automatically score generated answers for faithfulness and relevancy.
