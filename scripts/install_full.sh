#!/bin/bash

# Full installation script for Study Buddy with all dependencies

echo "==================================="
echo "Study Buddy Full Installation"
echo "==================================="

# Install basic requirements first
echo "📦 Installing basic requirements..."
pip3 install --user -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Basic requirements installed"
else
    echo "❌ Failed to install basic requirements"
    exit 1
fi

# Install full features
echo "📦 Installing full features (this may take a while)..."
pip3 install --user \
    "langchain>=0.1.0" \
    "langchain-community>=0.0.10" \
    "chromadb>=0.4.20" \
    "sentence-transformers>=2.2.0" \
    "pypdf2>=3.0.0" \
    "python-docx>=1.1.0" \
    "ollama>=0.1.0"

if [ $? -eq 0 ]; then
    echo "✅ Full features installed successfully"
else
    echo "⚠️  Some full features may not have installed. Basic functionality will still work."
fi

echo ""
echo "🧪 Testing installation..."
python3 test_installation.py

echo ""
echo "==================================="
echo "✅ Installation complete!"
echo "==================================="
echo ""
echo "🚀 Next steps:"
echo "1. Install Ollama: ./scripts/setup_ollama.sh"
echo "2. Try the demo: ./scripts/demo.sh"
echo "3. Or start using: python3 main.py --help"