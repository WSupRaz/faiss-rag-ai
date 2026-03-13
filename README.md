# FAISS RAG AI

This project implements a Retrieval-Augmented Generation (RAG) system using FAISS and Sentence Transformers.

## Features

- Reads PDF documents
- Splits text into chunks
- Converts chunks into embeddings
- Stores embeddings in FAISS vector database
- Performs semantic search
- Uses LLM to answer questions from the document

## Tech Stack

- Python
- FAISS
- Sentence Transformers
- PyPDF2
- OpenRouter API

## How It Works

PDF → Chunking → Embeddings → FAISS → Semantic Search → LLM Answer

## Run Project

```bash
pip install -r requirements.txt
python app.py
