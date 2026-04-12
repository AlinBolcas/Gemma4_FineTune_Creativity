from .evaluate import evaluate, export_eval, load_eval_prompts
from .gemma4_integration import load_gemma4, load_finetuned_gemma4

# Ollama is optional and usually unavailable on Kaggle / Colab.
try:
    from .ollama_integration import load_ollama_gemma4
except Exception:  # pragma: no cover - optional backend
    load_ollama_gemma4 = None
