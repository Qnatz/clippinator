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
    n_ctx: int = 4096  # Updated default
    n_threads: int = 4
    n_predict: int = 512  # Updated default
    temperature: float = 0.7  # Updated default
    top_k: int = 50  # Updated default
    top_p: float = 0.9  # Updated default
    repeat_penalty: float = 1.15  # Updated default
    n_gpu_layers: int = 0
    stop_sequences: Optional[List[str]] = None # Added
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
        if "LLAMA_CLI_STOP_SEQUENCES" in os.environ: # Added for stop_sequences
            # Assuming env var is a comma-separated string, e.g., "foo,bar"
            values_from_env["stop_sequences"] = os.environ["LLAMA_CLI_STOP_SEQUENCES"].split(',')


        # Combine with kwargs: kwargs override environment variables.
        # Pydantic will apply class-defined defaults for any keys not present in final_kwargs_to_pass.
        final_kwargs_to_pass = {**values_from_env, **kwargs}
        
        # ---- START DEBUG PRINTS ----
        # Ensure this print statement is still present for debugging this specific issue
        #print(f"DEBUG: CustomLlamaCliLLM __init__: final_kwargs_to_pass BEFORE super = {final_kwargs_to_pass}")
        # ---- END DEBUG PRINTS ----

        super().__init__(**final_kwargs_to_pass)
        
        # ---- START DEBUG PRINTS ----
        # Ensure this print statement is still present
        #print(f"DEBUG: CustomLlamaCliLLM __init__: self.__dict__ AFTER super().__init__ = {self.__dict__}")
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
            "stop_sequences": self.stop_sequences, # Added
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None, # This 'stop' is from Langchain's LLM._call signature
        run_manager: Optional[CallbackManagerForLLMRun] = None, # Added run_manager
        **kwargs: Any,
    ) -> str:
        # Use 'stop' from kwargs if provided during invocation, 
        # otherwise use 'stop' from _call signature (passed by Langchain),
        # otherwise use instance's 'self.stop_sequences'.
        effective_stop_sequences = kwargs.get('stop', stop or self.stop_sequences)

        # Add stop sequence awareness to the prompt (as per issue)
        # This part might be redundant if --stop is effective, but keeping for consistency with issue.
        # However, the original issue's example for _call adds this to the prompt *conditionally*.
        # Let's make it conditional as per the issue's CustomLlamaCliLLM._call.
        final_prompt = prompt
        if effective_stop_sequences:
            # The issue example prompt is "Instruction: Complete your response and end with one of these: {', '.join(stop)}"
            # The existing code adds "Important: You MUST stop generating further output when you encounter any of the following phrases: ..."
            # Let's use the one from the issue for this specific modification.
            final_prompt += f"\n\nInstruction: Complete your response and end with one of these: {', '.join(effective_stop_sequences)}"

        # Configure generation parameters (as per issue)
        generation_config = {
            "temperature": kwargs.get('temperature', self.temperature),
            "top_p": kwargs.get('top_p', self.top_p),
            "top_k": kwargs.get('top_k', self.top_k),
            "repeat_penalty": kwargs.get('repeat_penalty', self.repeat_penalty),
            "n_predict": kwargs.get('n_predict', self.n_predict),
            # "stop": effective_stop_sequences or [] # This was in the issue's generation_config but not used in command directly by name
        }

        # Build execution command (as per issue)
        command = [
            self.cli_path,
            "-m", self.model_path,
            "-p", final_prompt, # Use the potentially modified prompt
            "-c", str(self.n_ctx), # Added from existing functionality, seems important
            "-t", str(self.n_threads), # Added from existing functionality
            "--n-predict", str(generation_config["n_predict"]),
            "--temp", str(generation_config["temperature"]),
            "--top-p", str(generation_config["top_p"]),
            "--top-k", str(generation_config["top_k"]),
            "--repeat-penalty", str(generation_config["repeat_penalty"]),
            "--n-gpu-layers", str(self.n_gpu_layers) # Added from existing functionality
        ]
        
        # Add stop sequences if supported and available (as per issue)
        if effective_stop_sequences:
            command += ["--stop"] + list(effective_stop_sequences)
        
        logger.debug(f"Executing llama-cli command: {' '.join(command)}")

        # Execute and process output (as per issue)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=60  # Added timeout
            )
            output = result.stdout.strip()
            
            if result.stderr: # Log stderr even if process didn't fail
                logger.debug(f"llama-cli stderr: {result.stderr.strip()}")

            # Manual stop sequence truncation (as per issue)
            # This should ideally happen AFTER removing the prompt from the output.
            # The existing code already has some logic for removing the prompt from output. Let's try to integrate.
            
            # Attempt to remove the prompt (original or modified) from the output
            if output.startswith(final_prompt):
                output = output[len(final_prompt):].lstrip()
            elif output.startswith(prompt): # Fallback if only original prompt is found
                 output = output[len(prompt):].lstrip()

            for seq in effective_stop_sequences or []:
                if seq in output:
                    output = output.split(seq)[0]
                    break
                    
            return output
        except subprocess.TimeoutExpired:
            logger.warning("llama-cli generation timeout occurred")
            return "Generation timeout occurred" # As per issue
        except subprocess.CalledProcessError as e: # Keep existing error handling
            logger.error(f"llama-cli execution failed. STDOUT: {e.stdout.strip()} STDERR: {e.stderr.strip()}")
            raise ValueError(f"Error from llama-cli: {e.stderr.strip()}") from e
        except FileNotFoundError: 
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
