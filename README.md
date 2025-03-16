# History RAG - Vietnamese History QA System

## Overview
History RAG is an intelligent question-answering system for Vietnamese history that combines Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs). It enables users to ask questions in Vietnamese and receive detailed, accurate answers based on reliable historical sources.

## Features
- Hybrid search combining BM25 and Vector Search
- Vietnamese language processing with PhoBERT
- Intelligent answer generation using LangChain and Google's Gemini
- Source references and confidence scores

## Architecture
The system consists of three main components:
1. **Retrieval System**: Hybrid search combining semantic similarity (Milvus) and keyword matching (BM25)
2. **Question-Answering Chain**: LangChain-based pipeline using Gemini
3. **Web Interface**: Streamlit-based UI

## Prerequisites
- Python 3.8+
- MongoDB
- Milvus Vector Database
- Google API Key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/history-rag.git
cd history-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Add your Google API key to .env file
```

4. Start the databases:
```bash
# Start MongoDB
mongod

# Start Milvus
docker-compose up -d
```

5. Run the application:
```bash
streamlit run app.py
```

## Project Structure
```
history-rag/
├── app.py                 # Streamlit application
├── config/               
│   └── config.yaml       # System configuration
├── src/
│   ├── data/             # Data processing and loading
│   │   ├── data_loader.py
│   │   └── data_processor.py
│   ├── database/
│   │   ├── mongo_client.py
│   │   └── vector_store.py        # Database connections
│   ├── llm/              # Language model components
│   │   └── chain.py
│   └── rag/              # RAG pipeline
│       ├── embeddings.py
│       ├── query_transformer.py
│       └── retriever.py
└── requirements.txt
```


## Processing Pipeline
1. **Query Processing**:
   - Transform questions into search-optimized format
   - Generate query embeddings

2. **Retrieval**:
   - Semantic search using Milvus
   - Keyword search using BM25
   - Combine and rank results

3. **Answer Generation**:
   - Format context from documents
   - Generate LLM prompt
   - Generate answer

