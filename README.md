# Study-Buddy 📚
A lightweight offline study assistant powered by Retrieval-Augmented Generation (RAG). Combines ChromaDB vector storage, HuggingFace embeddings, and Ollama local LLMs for intelligent document-based question answering.

## ✨ Features

- **🔒 Fully Offline**: No internet required after setup
- **📄 Multi-format Support**: PDF and TXT document ingestion
- **🧠 Smart Retrieval**: ChromaDB vector storage with semantic search
- **🤖 Local LLM**: Ollama integration (Mistral, LLaMA2, etc.)
- **🚀 Easy CLI**: Simple command-line interface
- **📊 Rich Output**: Beautiful terminal interface with Rich
- **🏗️ Modular Design**: Clean, extensible architecture

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com) installed and running

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KoalaLyfestyle/Study-Buddy.git
   cd Study-Buddy
   ```

2. **Run the setup script**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. **Start Ollama and pull a model**
   ```bash
   ollama serve
   ollama pull mistral  # or llama2, codellama, etc.
   ```

### Basic Usage

1. **Ingest documents**
   ```bash
   python3 main.py ingest examples/
   ```

2. **Ask questions**
   ```bash
   python3 main.py query "What is machine learning?"
   ```

3. **Search documents**
   ```bash
   python3 main.py search "Python programming"
   ```

4. **Interactive mode**
   ```bash
   python3 main.py interactive
   ```

## 📋 Commands

| Command | Description | Example |
|---------|-------------|---------|
| `ingest` | Add documents to knowledge base | `python3 main.py ingest documents/` |
| `query` | Ask AI questions about documents | `python3 main.py query "What is Python?"` |
| `search` | Find relevant document passages | `python3 main.py search "machine learning"` |
| `status` | Show system and knowledge base info | `python3 main.py status` |
| `clear` | Clear all documents from knowledge base | `python3 main.py clear` |
| `interactive` | Start interactive Q&A session | `python3 main.py interactive` |

## 🛠️ Configuration

Edit `src/study_buddy/config.py` to customize:

- **Embedding Model**: Change HuggingFace model (default: `all-MiniLM-L6-v2`)
- **Chunk Size**: Adjust document chunking (default: 1000 characters)
- **Ollama Settings**: Configure URL and default model
- **Paths**: Customize data directories

## 📁 Project Structure

```
Study-Buddy/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── src/study_buddy/        # Core package
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── document_processor.py  # Document loading and processing
│   ├── vector_store.py    # ChromaDB vector storage
│   ├── llm_manager.py     # Ollama LLM integration
│   ├── rag_assistant.py   # Main RAG orchestrator
│   └── cli.py             # Command-line interface
├── data/                  # Data storage
│   ├── documents/         # Original documents
│   └── vector_store/      # ChromaDB database
├── examples/              # Sample documents
├── scripts/               # Utility scripts
│   ├── setup.sh          # Setup script
│   └── demo.sh           # Demo script
```

## 🔧 Advanced Usage

### Custom Models

Use different Ollama models:
```bash
python3 main.py --model llama2 query "Explain neural networks"
```

### Batch Processing

Ingest multiple directories:
```bash
python3 main.py ingest papers/
python3 main.py ingest textbooks/
python3 main.py ingest notes/
```

### Configuration via Environment

Create a `.env` file:
```env
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=1500
DEFAULT_MODEL=codellama
OLLAMA_BASE_URL=http://localhost:11434
```

## 📊 Dependencies

### Core Libraries
- **LangChain**: RAG framework and document processing
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: HuggingFace embeddings
- **Ollama**: Local LLM inference

### Document Processing
- **PyPDF2**: PDF document parsing
- **python-docx**: Word document support (future)

### CLI & UI
- **Click**: Command-line interface framework
- **Rich**: Beautiful terminal output

## 🎯 Use Cases

- **📖 Study Materials**: Index textbooks, papers, and notes
- **📝 Research**: Query large document collections
- **💻 Code Documentation**: Search through technical docs
- **🎓 Exam Prep**: Quiz yourself on course materials
- **📊 Knowledge Management**: Personal knowledge base

## 🔍 Example Queries

```bash
# General questions
python3 main.py query "What are the main types of machine learning?"

# Specific searches
python3 main.py search "Python data structures"

# Complex questions
python3 main.py query "Compare supervised and unsupervised learning with examples"
```

## 🚨 Troubleshooting

### Ollama Issues
- **Server not running**: `ollama serve`
- **Model not found**: `ollama pull mistral`
- **Connection error**: Check `http://localhost:11434`

### Performance Tips
- Use smaller embedding models for faster processing
- Adjust chunk size based on document types
- Consider GPU acceleration for large document sets

### Common Errors
- **Import errors**: Ensure all dependencies are installed
- **Permission errors**: Check file permissions on data directories
- **Memory issues**: Reduce chunk size or use smaller models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for RAG framework
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [Ollama](https://ollama.com) for local LLM inference
- [HuggingFace](https://huggingface.co) for embedding models

---

**Happy Learning! 🎓✨**
