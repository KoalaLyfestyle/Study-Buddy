#!/usr/bin/env python3
"""
Test script to verify Study Buddy installation and functionality.
"""

import sys
import logging
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from study_buddy.config import config
        print("✅ Config module imported successfully")
        
        from study_buddy.document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
        
        from study_buddy.vector_store import VectorStore
        print("✅ VectorStore imported successfully")
        
        from study_buddy.llm_manager import LLMManager
        print("✅ LLMManager imported successfully")
        
        from study_buddy.rag_assistant import RAGAssistant
        print("✅ RAGAssistant imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    print("\nTesting basic functionality...")
    try:
        from study_buddy.rag_assistant import RAGAssistant
        
        # Initialize assistant
        assistant = RAGAssistant()
        print("✅ RAG Assistant initialized")
        
        # Test status
        status = assistant.get_knowledge_base_info()
        print(f"✅ Status retrieved: {status.get('vector_store', {}).get('document_count', 0)} documents")
        
        return True
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_cli():
    """Test CLI functionality."""
    print("\nTesting CLI...")
    try:
        from study_buddy.cli import cli
        print("✅ CLI module imported successfully")
        return True
    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False

def main():
    """Run all tests."""
    print("Study Buddy Installation Test")
    print("=" * 40)
    
    # Suppress logging for cleaner output
    logging.getLogger().setLevel(logging.ERROR)
    
    tests = [
        ("Import Test", test_imports),
        ("Functionality Test", test_basic_functionality),
        ("CLI Test", test_cli),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("Test Results:")
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Study Buddy is ready to use.")
        print("\nNext steps:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull a model: ollama pull mistral")
        print("3. Ingest documents: python3 main.py ingest examples/")
        print("4. Ask questions: python3 main.py query 'What is machine learning?'")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()