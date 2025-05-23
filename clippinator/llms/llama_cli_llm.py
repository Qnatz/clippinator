from __future__ import annotations

import os
import subprocess
import logging
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

logger = logging.getLogger(__name__)

class CustomLlamaCliLLM(LLM):
    """
    Custom LLM class to wrap the llama-cli executable.
    """

    cli_path: str
    model_path: str
    n_ctx: int = 2048
    n_threads: int = 4
    n_predict: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    n_gpu_layers: int = 0
    # Add any other parameters you plan to make configurable via __init__

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs) # Pydantic will handle assignments based on class annotations

        # Read from environment variables and override defaults if set
        self.cli_path = os.environ.get("LLAMA_CLI_PATH", self.cli_path) # Default should be None or error if not set
        if not self.cli_path:
            raise ValueError("LLAMA_CLI_PATH environment variable not set.")

        self.model_path = os.environ.get("MODEL_PATH", self.model_path) # Default should be None or error if not set
        if not self.model_path:
            raise ValueError("MODEL_PATH environment variable not set.")

        self.n_ctx = int(os.environ.get("LLAMA_CLI_N_CTX", self.n_ctx))
        self.n_threads = int(os.environ.get("LLAMA_CLI_N_THREADS", self.n_threads))
        self.n_predict = int(os.environ.get("LLAMA_CLI_N_PREDICT", self.n_predict))
        self.temperature = float(os.environ.get("LLAMA_CLI_TEMPERATURE", self.temperature))
        self.top_k = int(os.environ.get("LLAMA_CLI_TOP_K", self.top_k))
        self.top_p = float(os.environ.get("LLAMA_CLI_TOP_P", self.top_p))
        self.repeat_penalty = float(os.environ.get("LLAMA_CLI_REPEAT_PENALTY", self.repeat_penalty))
        self.n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", self.n_gpu_layers)) # Using N_GPU_LAYERS as decided

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
        stop: Optional[List[str]] = None, # stop sequences are not easily supported by llama-cli in a single run.
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            logger.warning(f"Stop sequences {stop} are not directly supported by llama-cli wrapper and will be ignored.")

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
            "--n-gpu-layers", str(self.n_gpu_layers) # Ensure this flag is correct for llama-cli
        ]
        
        # Add other llama-cli specific flags if needed based on parameters
        # For example, you might need to add flags like --no-mlock if self.use_mlock = False, etc.
        # For now, keeping it to the parameters defined.

        logger.debug(f"Executing llama-cli command: {' '.join(command)}")

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True  # Will raise CalledProcessError if llama-cli exits with non-zero
            )
            
            generated_text = process.stdout.strip()
            # llama-cli might print the prompt first, then the generation.
            # A common behavior is that the generation starts right after the prompt.
            # This parsing might need to be made more robust.
            if generated_text.startswith(prompt):
                 #This simple removal might not be robust enough if prompt has special characters or is very long
                 #A more robust way would be to find the end of the prompt in the output, but that's complex.
                 #For now, let's assume the output starts with the prompt.
                generated_text = generated_text[len(prompt):].lstrip() # Remove prompt and leading space
            
            logger.debug(f"llama-cli stderr: {process.stderr.strip()}")
            return generated_text

        except subprocess.CalledProcessError as e:
            logger.error(f"llama-cli execution failed with error: {e.stderr}")
            return f"Error from llama-cli: {e.stderr}" # Or raise an exception
        except FileNotFoundError:
            logger.error(f"LLAMA_CLI_PATH not found at {self.cli_path}. Please ensure it's correctly set.")
            return "Error: llama-cli executable not found." # Or raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while running llama-cli: {e}")
            return f"Unexpected error: {e}" # Or raise

# Example of how it might be instantiated (for testing, not part of the class itself)
if __name__ == '__main__':
    # Set environment variables for testing before running this part
    # export LLAMA_CLI_PATH="/path/to/your/llama-cli"
    # export MODEL_PATH="/path/to/your/model.gguf"
    if os.environ.get("LLAMA_CLI_PATH") and os.environ.get("MODEL_PATH"):
        try:
            llm = CustomLlamaCliLLM()
            test_prompt = "Explain the importance of bees in 100 words."
            print(f"Sending prompt to llama-cli: '{test_prompt}'")
            response = llm(test_prompt)
            print(f"Response from llama-cli:\n{response}")
        except ValueError as ve:
            print(ve)
    else:
        print("Please set LLAMA_CLI_PATH and MODEL_PATH environment variables to test CustomLlamaCliLLM.")
