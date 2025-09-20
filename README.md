# Study Buddy

A local AI study assistant that helps you interact with your notes using retrieval-augmented generation (RAG). Ask questions about your study materials, generate quiz questions, or get summaries.

## Features

- Search through your study notes with natural language queries
- Generate practice questions on specific topics
- Summarize text content
- Local LLM processing using Ollama (no API keys required)
- Web interface built with Streamlit

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install and start Ollama with the Mistral model:

   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull mistral
   ```

3. Add your study materials to the `data/` directory as text files

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

Once running, you can:

- Ask questions about your notes: "What are the key concepts in machine learning?"
- Request quiz questions: "Give me practice questions on linear algebra"
- Get summaries: "Summarize this chapter on neural networks"

The system uses a vector database to find relevant content from your notes and an AI agent to provide helpful responses.

## Architecture

- `app.py` - Streamlit web interface
- `agent/` - LangChain agent and tools for note search, quiz generation, and summarization
- `rag.py` - Vector store and retrieval system using LlamaIndex
- `config.py` - LLM configuration for local Ollama models
- `data/` - Directory for your study materials

## Requirements

- Python 3.8+
- Ollama running locally
- At least 8GB RAM recommended for running local models

## Acknowledgements

This project uses several open-source tools:

- **LangChain** - AI agent framework and LLM orchestration
- **LlamaIndex** - Vector store and document retrieval system
- **Streamlit** - Web application framework
- **Ollama** - Local LLM runtime (runs Mistral model)
- **Hugging Face** - Text embedding models for semantic search
