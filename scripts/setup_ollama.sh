#!/bin/bash

# Ollama installation and setup script for Study Buddy

echo "==================================="
echo "Ollama Setup for Study Buddy"
echo "==================================="

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed: $(ollama --version 2>/dev/null || echo 'version check failed')"
else
    echo "📦 Installing Ollama..."
    
    # Install Ollama
    if command -v curl &> /dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh
        
        if [ $? -eq 0 ]; then
            echo "✅ Ollama installed successfully"
        else
            echo "❌ Failed to install Ollama"
            exit 1
        fi
    else
        echo "❌ curl is not installed. Please install curl first."
        exit 1
    fi
fi

# Check if Ollama service is running
echo "🔍 Checking Ollama service status..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama server is running"
else
    echo "⚠️  Ollama server is not running. Starting it..."
    echo "💡 You may need to run 'ollama serve' in a separate terminal"
    
    # Try to start Ollama in background
    if command -v ollama &> /dev/null; then
        echo "Starting Ollama server in background..."
        nohup ollama serve > /dev/null 2>&1 &
        sleep 5
        
        # Check again
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama server started successfully"
        else
            echo "⚠️  Please start Ollama manually with: ollama serve"
        fi
    fi
fi

# List available models
echo "📋 Checking available models..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    available_models=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -v "^$")
    
    if [ -z "$available_models" ]; then
        echo "📥 No models found. Downloading recommended models..."
        
        echo "⬇️  Downloading mistral (recommended for general use)..."
        ollama pull mistral
        
        if [ $? -eq 0 ]; then
            echo "✅ Mistral model downloaded successfully"
        else
            echo "❌ Failed to download Mistral model"
        fi
        
        echo ""
        echo "💡 Other recommended models:"
        echo "   ollama pull llama2        # Alternative general-purpose model"
        echo "   ollama pull codellama     # Better for programming questions"
        echo "   ollama pull llama2:7b     # Smaller, faster model"
    else
        echo "✅ Available models:"
        echo "$available_models" | sed 's/^/   /'
    fi
else
    echo "⚠️  Cannot check models. Make sure Ollama server is running."
fi

echo ""
echo "==================================="
echo "✅ Ollama setup complete!"
echo "==================================="
echo ""
echo "🚀 Ready to use Study Buddy with Ollama:"
echo "1. python3 main.py ingest examples/"
echo "2. python3 main.py query 'What is machine learning?'"
echo ""
echo "📖 For more commands: python3 main.py --help"