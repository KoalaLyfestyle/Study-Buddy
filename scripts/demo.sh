#!/bin/bash

# Quick demo script for Study Buddy RAG Assistant

echo "==================================="
echo "Study Buddy RAG Assistant Demo"
echo "==================================="

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found. Please run this script from the project root."
    exit 1
fi

echo "📚 Ingesting example documents..."
python3 main.py ingest examples/

echo ""
echo "ℹ️  Checking system status..."
python3 main.py status

echo ""
echo "❓ Asking a sample question..."
python3 main.py query "What is machine learning?"

echo ""
echo "🔍 Searching for Python-related content..."
python3 main.py search "Python programming"

echo ""
echo "==================================="
echo "✅ Demo complete!"
echo "==================================="
echo ""
echo "🎯 Try these commands:"
echo "  python3 main.py interactive    # Start interactive mode"
echo "  python3 main.py query 'your question here'"
echo "  python3 main.py search 'search term'"
echo "  python3 main.py status         # Check system status"