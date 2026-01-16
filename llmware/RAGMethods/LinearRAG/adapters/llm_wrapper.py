"""LLM wrappers for LinearRAG.

Provides adapters to bridge LinearRAG's chat-style inference interface
with llmware's prompt-based inference interface.
"""

from typing import List, Dict, Protocol, runtime_checkable, Any
import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM models used by LinearRAG.
    
    Any LLM must implement this interface to work with LinearRAG's QA pipeline.
    The key method is `infer` which takes chat-style messages and returns a response.
    """
    
    def infer(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from chat messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Roles are typically 'system', 'user', 'assistant'.
                      
        Returns:
            Generated response string
        """
        ...


class LLMWareModelWrapper:
    """Wrapper for llmware models to provide chat-style inference.
    
    Bridges the interface gap between LinearRAG's expected `.infer(messages)` 
    and llmware's `.inference(prompt)` interface.
    
    Args:
        llmware_model: An llmware model loaded via ModelCatalog
        max_output: Maximum output tokens (default: 200)
        
    Example:
        >>> from llmware.models import ModelCatalog
        >>> llm = ModelCatalog().load_model("llmware/bling-phi-3-gguf")
        >>> wrapper = LLMWareModelWrapper(llm)
        >>> response = wrapper.infer([
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "What is 2+2?"}
        ... ])
    """
    
    def __init__(self, llmware_model: Any, max_output: int = 200):
        self.model = llmware_model
        self.max_output = max_output
        
        # Check that model has inference method
        if not hasattr(llmware_model, 'inference'):
            raise ValueError(
                "llmware model must have an 'inference' method. "
                "Make sure you're using a generative model."
            )
        
        model_name = getattr(llmware_model, 'model_name', 'unknown')
        logger.info(f"Wrapped llmware LLM model: {model_name}")
    
    def infer(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from chat messages using llmware model.
        
        Converts chat-style messages to a prompt format suitable for
        llmware's inference method.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Generated response string
        """
        # Extract system and user content from messages
        system_content = ""
        user_content = ""
        context_content = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content
            elif role == "assistant":
                # Previous assistant messages could be added as context
                context_content += content + "\n"
        
        # Call llmware model inference
        try:
            response = self.model.inference(
                prompt=user_content,
                add_context=context_content if context_content else None,
                add_prompt_engineering=system_content if system_content else None
            )
            
            # Handle different response formats from llmware
            if isinstance(response, dict):
                # Most llmware models return a dict with 'llm_response' key
                return response.get("llm_response", str(response))
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            raise


class OpenAILLMAdapter:
    """Adapter for OpenAI API models.
    
    Provides direct OpenAI API access for LinearRAG's QA pipeline.
    
    Args:
        model_name: OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        base_url: Optional custom base URL for API
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
        
    Example:
        >>> adapter = OpenAILLMAdapter("gpt-4")
        >>> response = adapter.infer([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str = None,
        base_url: str = None,
        max_tokens: int = 2000,
        temperature: float = 0
    ):
        import os
        
        try:
            from openai import OpenAI
            import httpx
        except ImportError:
            raise ImportError(
                "openai and httpx are required for OpenAI adapter. "
                "Install with: pip install openai httpx"
            )
        
        self.model_name = model_name
        
        http_client = httpx.Client(timeout=60.0, trust_env=False)
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            http_client=http_client
        )
        
        self.config = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        logger.info(f"Initialized OpenAI adapter with model: {model_name}")
    
    def infer(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI API.
        
        Args:
            messages: Chat messages in OpenAI format
            
        Returns:
            Generated response string
        """
        response = self.client.chat.completions.create(
            **self.config,
            messages=messages
        )
        return response.choices[0].message.content


class HuggingFaceLLMAdapter:
    """Adapter for local HuggingFace models.
    
    Loads and runs HuggingFace models locally for LinearRAG's QA pipeline.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to use ('cuda', 'cpu', or 'auto')
        max_new_tokens: Maximum new tokens to generate
        
    Example:
        >>> adapter = HuggingFaceLLMAdapter("Qwen/Qwen2.5-1.5B-Instruct")
        >>> response = adapter.infer([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 128
    ):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for HuggingFace adapter. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading HuggingFace model {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model {model_name} loaded successfully")
    
    def infer(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using local HuggingFace model.
        
        Args:
            messages: Chat messages
            
        Returns:
            Generated response string
        """
        import torch
        
        # Try to use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)
        else:
            # Fallback: construct prompt manually
            system_message = ""
            user_message = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "user":
                    user_message = msg["content"]
            
            if system_message:
                prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        return response.strip()

