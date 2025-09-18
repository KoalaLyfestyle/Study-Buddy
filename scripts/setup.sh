#!/bin/bash

# Study Buddy Setup Script
# This script helps you set up the Study Buddy RAG Assistant

echo "==================================="
echo "Study Buddy RAG Assistant Setup"
echo "==================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 found"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama is not installed."
    echo "📋 To install Ollama, run:"
    echo "   curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo "📋 After installation, start Ollama and pull a model:"
    echo "   ollama serve"
    echo "   ollama pull mistral"
    echo ""
else
    echo "✅ Ollama found: $(ollama --version 2>/dev/null || echo 'version check failed')"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama server is running"
        
        # List available models
        echo "📋 Available models:"
        ollama list 2>/dev/null || echo "   No models found. Consider running: ollama pull mistral"
    else
        echo "⚠️  Ollama server is not running."
        echo "📋 Start Ollama with: ollama serve"
    fi
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/documents data/vector_store

echo ""
echo "==================================="
echo "✅ Setup complete!"
echo "==================================="
echo ""
echo "🚀 Quick start:"
echo "1. Place your documents (PDF/TXT) in the 'examples/' directory"
echo "2. Run: python3 main.py ingest examples/"
echo "3. Run: python3 main.py query 'What is machine learning?'"
echo ""
echo "📖 For more help: python3 main.py --help"