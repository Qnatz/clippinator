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
        # Precedence for setting values:
        # 1. Keyword arguments passed to __init__ (kwargs)
        # 2. Environment variables
        # 3. Class-defined defaults (handled by Pydantic when a key is not in final_kwargs_to_pass)

        values_from_env = {}

        # Populate from environment variables, converting types.
        if "LLAMA_CLI_PATH" in os.environ:
            values_from_env["cli_path"] = os.environ["LLAMA_CLI_PATH"]
        if "MODEL_PATH" in os.environ:
            values_from_env["model_path"] = os.environ["MODEL_PATH"]
        if "LLAMA_CLI_N_CTX" in os.environ:
            values_from_env["n_ctx"] = int(os.environ["LLAMA_CLI_N_CTX"])
        if "LLAMA_CLI_N_THREADS" in os.environ:
            values_from_env["n_threads"] = int(os.environ["LLAMA_CLI_N_THREADS"])
        if "LLAMA_CLI_N_PREDICT" in os.environ:
            values_from_env["n_predict"] = int(os.environ["LLAMA_CLI_N_PREDICT"])
        if "LLAMA_CLI_TEMPERATURE" in os.environ:
            values_from_env["temperature"] = float(os.environ["LLAMA_CLI_TEMPERATURE"])
        if "LLAMA_CLI_TOP_K" in os.environ:
            values_from_env["top_k"] = int(os.environ["LLAMA_CLI_TOP_K"])
        if "LLAMA_CLI_TOP_P" in os.environ:
            values_from_env["top_p"] = float(os.environ["LLAMA_CLI_TOP_P"])
        if "LLAMA_CLI_REPEAT_PENALTY" in os.environ:
            values_from_env["repeat_penalty"] = float(os.environ["LLAMA_CLI_REPEAT_PENALTY"])
        if "N_GPU_LAYERS" in os.environ: # Using N_GPU_LAYERS as per previous definition
            values_from_env["n_gpu_layers"] = int(os.environ["N_GPU_LAYERS"])

        # Combine with kwargs: kwargs override environment variables.
        # Pydantic will apply class-defined defaults for any keys not present in final_kwargs_to_pass.
        final_kwargs_to_pass = {**values_from_env, **kwargs}
        
        # ---- START DEBUG PRINTS ----
        # Ensure this print statement is still present for debugging this specific issue
        print(f"DEBUG: CustomLlamaCliLLM __init__: final_kwargs_to_pass BEFORE super = {final_kwargs_to_pass}")
        # ---- END DEBUG PRINTS ----

        super().__init__(**final_kwargs_to_pass)
        
        # ---- START DEBUG PRINTS ----
        # Ensure this print statement is still present
        print(f"DEBUG: CustomLlamaCliLLM __init__: self.__dict__ AFTER super().__init__ = {self.__dict__}")
        # ---- END DEBUG PRINTS ----
        # Validators for cli_path and model_path will run after Pydantic initializes fields based on final_kwargs_to_pass and class defaults.

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
