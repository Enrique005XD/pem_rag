# Hybrid RAG System for PCI DSS Knowledge

## Project Overview
This project implements a **Retrieval-Augmented Generation (RAG) system** specialized for **PCI DSS documentation**. It combines multiple AI models and tools to provide accurate, structured, and readable answers for compliance-related queries.  

Key components:

- **Retriever**: Extracts top-K relevant chunks from PCI DSS PDFs using FAISS vector indexing.
- **Mock LLM (Gemini API)**: Generates preliminary summaries and refines outputs for clean formatting.
- **SLM (Gemma 2B via Ollama)**: Local fallback model for offline reasoning and deterministic responses.
- **Router**: Dynamically selects between LLM and SLM based on confidence, performance, and context.
- **Google Search Tool**: Provides live search results when the local corpus is insufficient.

The system ensures:
- Non-hallucinated, factual responses.
- Structured output with concise summaries and bullet points.
- Multi-layered fallback strategy to guarantee answer delivery.

---

## File Structure

