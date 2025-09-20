from langchain_ollama import OllamaLLM

def get_llm(model: str = "mistral") -> OllamaLLM:
    """Return a local LLM from Ollama (default: mistral) with optimized settings."""
    return OllamaLLM(
        model=model,
        temperature=0.1,
        timeout=120,
        num_predict=1000
    )