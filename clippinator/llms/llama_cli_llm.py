from __future__ import annotations

import os
import subprocess
import logging
from typing import Any, List, Mapping, Optional, Dict

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
# For Pydantic v1 style validators
from pydantic.v1 import root_validator, validator 

logger = logging.getLogger(__name__)

class CustomLlamaCliLLM(LLM):
    """
    Custom LLM class to wrap the llama-cli executable.
    """

    cli_path: Optional[str] = None
    model_path: Optional[str] = None
    n_ctx: int = 2048
    n_threads: int = 4
    n_predict: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    n_gpu_layers: int = 0
    # Add any other parameters that are meant to be configurable

    # Pydantic v1 style root_validator to ensure essential paths are loaded from env vars
    # This validator runs after __init__ assignments if values are passed via kwargs,
    # but we want to load from env vars if not passed via kwargs.
    # A more robust way for Pydantic v1 is to handle this in __init__ before super()
    # or ensure they are passed to super().
    # Let's refine __init__ and use validators for final checks.

    def __init__(self, **kwargs: Any):
        # Prioritize kwargs passed during instantiation
        _cli_path = kwargs.pop("cli_path", os.environ.get("LLAMA_CLI_PATH"))
        _model_path = kwargs.pop("model_path", os.environ.get("MODEL_PATH"))
        
        # For other params, allow kwargs to override env vars, which override class defaults
        _n_ctx = int(kwargs.pop("n_ctx", os.environ.get("LLAMA_CLI_N_CTX", self.n_ctx)))
        _n_threads = int(kwargs.pop("n_threads", os.environ.get("LLAMA_CLI_N_THREADS", self.n_threads)))
        _n_predict = int(kwargs.pop("n_predict", os.environ.get("LLAMA_CLI_N_PREDICT", self.n_predict)))
        _temperature = float(kwargs.pop("temperature", os.environ.get("LLAMA_CLI_TEMPERATURE", self.temperature)))
        _top_k = int(kwargs.pop("top_k", os.environ.get("LLAMA_CLI_TOP_K", self.top_k)))
        _top_p = float(kwargs.pop("top_p", os.environ.get("LLAMA_CLI_TOP_P", self.top_p)))
        _repeat_penalty = float(kwargs.pop("repeat_penalty", os.environ.get("LLAMA_CLI_REPEAT_PENALTY", self.repeat_penalty)))
        _n_gpu_layers = int(kwargs.pop("n_gpu_layers", os.environ.get("N_GPU_LAYERS", self.n_gpu_layers)))

        # Update kwargs with resolved values to pass to super().__init__
        # This ensures Pydantic knows about them during its initialization.
        resolved_kwargs = {
            "cli_path": _cli_path,
            "model_path": _model_path,
            "n_ctx": _n_ctx,
            "n_threads": _n_threads,
            "n_predict": _n_predict,
            "temperature": _temperature,
            "top_k": _top_k,
            "top_p": _top_p,
            "repeat_penalty": _repeat_penalty,
            "n_gpu_layers": _n_gpu_layers,
            **kwargs # Add back any remaining kwargs
        }
        super().__init__(**resolved_kwargs)

    @validator('cli_path', always=True)
    def validate_cli_path(cls, v):
        if not v:
            raise ValueError("LLAMA_CLI_PATH environment variable not set or cli_path not provided.")
        if not os.path.exists(v):
            raise FileNotFoundError(f"LLAMA_CLI_PATH executable not found at {v}")
        if not os.access(v, os.X_OK):
            raise PermissionError(f"LLAMA_CLI_PATH executable at {v} is not executable.")
        return v

    @validator('model_path', always=True)
    def validate_model_path(cls, v):
        if not v:
            raise ValueError("MODEL_PATH environment variable not set or model_path not provided.")
        if not os.path.exists(v):
            raise FileNotFoundError(f"Model file not found at {v}")
        return v
        
    @property
    def _llm_type(self) -> str:
        return "custom_llama_cli"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "cli_path": self.cli_path,
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "n_gpu_layers": self.n_gpu_layers,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            logger.warning(f"Stop sequences {stop} are not directly supported by llama-cli wrapper and will be ignored.")

        # Ensure cli_path and model_path are validated and available
        if not self.cli_path or not self.model_path:
             # This should ideally be caught by validators, but as a safeguard:
            raise ValueError("cli_path or model_path is not configured properly.")

        command = [
            self.cli_path,
            "-m", self.model_path,
            "-p", prompt,
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "--n-predict", str(self.n_predict),
            "--temp", str(self.temperature),
            "--top-k", str(self.top_k),
            "--top-p", str(self.top_p),
            "--repeat-penalty", str(self.repeat_penalty),
            "--n-gpu-layers", str(self.n_gpu_layers)
        ]
        
        logger.debug(f"Executing llama-cli command: {' '.join(command)}")

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            generated_text = process.stdout.strip()
            if generated_text.startswith(prompt): # Basic parsing, might need refinement
                generated_text = generated_text[len(prompt):].lstrip() 
            
            if process.stderr: # Log stderr even if process didn't fail
                logger.debug(f"llama-cli stderr: {process.stderr.strip()}")
            return generated_text

        except subprocess.CalledProcessError as e:
            logger.error(f"llama-cli execution failed. STDOUT: {e.stdout.strip()} STDERR: {e.stderr.strip()}")
            # Consider whether to return stderr or raise a more specific exception
            raise ValueError(f"Error from llama-cli: {e.stderr.strip()}") from e
        except FileNotFoundError: # Should be caught by validator, but good practice
            logger.error(f"LLAMA_CLI_PATH not found at {self.cli_path}.")
            raise FileNotFoundError(f"llama-cli executable not found at {self.cli_path}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while running llama-cli: {e}")
            raise RuntimeError(f"Unexpected error running llama-cli: {e}") from e

# Keep the if __name__ == '__main__': block for testing as is.
# Example of how it might be instantiated (for testing, not part of the class itself)
if __name__ == '__main__':
    # Set environment variables for testing before running this part
    # export LLAMA_CLI_PATH="/path/to/your/llama-cli"
    # export MODEL_PATH="/path/to/your/model.gguf"
    if os.environ.get("LLAMA_CLI_PATH") and os.environ.get("MODEL_PATH"):
        try:
            # Test with explicit params (would override env vars or defaults)
            # llm = CustomLlamaCliLLM(n_ctx=4096) 
            llm = CustomLlamaCliLLM() # Test with env vars and defaults
            test_prompt = "Explain the importance of bees in 100 words."
            print(f"Sending prompt to llama-cli: '{test_prompt}'")
            response = llm(test_prompt)
            print(f"Response from llama-cli:\n{response}")
        except Exception as e: # Catch any exception during init or call
            print(f"Error during test: {e}")
    else:
        print("Please set LLAMA_CLI_PATH and MODEL_PATH environment variables to test CustomLlamaCliLLM.")
