from typing import List, Dict

from .base import Generator


class LLMGenerator(Generator):
    """LLM generator using llmware models."""
    
    def __init__(self, model_name: str = "llmware/bling-1b-0.1"):
        """
        Initialize LLM generator.
        
        Args:
            model_name: Name of the llmware model to use
        """
        super().__init__()
        self.model_name = model_name
        try:
            from llmware.models import ModelCatalog
            self.model = ModelCatalog().load_model(model_name)
        except Exception as e:
            raise ImportError(f"Failed to load model {model_name}: {e}")
    
    def generate(
        self,
        query: str,
        context: List[Dict],
    ) -> str:
        """Generate response using LLM with RAG context."""
        # Combine context into prompt
        context_text = "\n\n".join([doc.get("text", "") for doc in context])
        
        # Create prompt
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response
        response = self.model.inference(prompt)
        
        # Extract text from response
        if isinstance(response, dict):
            return response.get("llm_response", response.get("text", str(response)))
        elif isinstance(response, str):
            return response
        else:
            return str(response)
