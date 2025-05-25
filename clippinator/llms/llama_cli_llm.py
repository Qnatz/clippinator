from __future__ import annotations

import os
import subprocess
import logging
from typing import Any, List, Mapping, Optional, Dict

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic.v1 import root_validator, validator 

logger = logging.getLogger(__name__)

class CustomLlamaCliLLM(LLM):
    """Custom LLM wrapper for llama-cli executable with enhanced environment handling"""

    cli_path: Optional[str] = None
    model_path: Optional[str] = None
    n_ctx: int = 4096
    n_threads: int = 4
    n_predict: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repeat_penalty: float = 1.15
    n_gpu_layers: int = 0
    stop_sequences: Optional[List[str]] = None

    def __init__(self, **kwargs: Any):
        """Initialize with proper configuration merging"""
        values_from_env = self._get_values_from_env()
        final_kwargs = self._merge_config_sources(values_from_env, kwargs)
        super().__init__(**final_kwargs)

    def _get_values_from_env(self) -> Dict[str, Any]:
        """Extract and convert environment variables"""
        env_mappings = {
            'LLAMA_CLI_PATH': ('cli_path', str),
            'MODEL_PATH': ('model_path', str),
            'LLAMA_CLI_N_CTX': ('n_ctx', int),
            'LLAMA_CLI_N_THREADS': ('n_threads', int),
            'LLAMA_CLI_N_PREDICT': ('n_predict', int),
            'LLAMA_CLI_TEMPERATURE': ('temperature', float),
            'LLAMA_CLI_TOP_K': ('top_k', int),
            'LLAMA_CLI_TOP_P': ('top_p', float),
            'LLAMA_CLI_REPEAT_PENALTY': ('repeat_penalty', float),
            'N_GPU_LAYERS': ('n_gpu_layers', int),
            'LLAMA_CLI_STOP_SEQUENCES': ('stop_sequences', lambda x: x.split(',')),
        }

        values = {}
        for env_var, (field, converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    values[field] = converter(os.environ[env_var])
                except Exception as e:
                    logger.warning(f"Error converting {env_var}: {e}")
        return values

    def _merge_config_sources(self, env_vars: Dict, kwargs: Dict) -> Dict:
        """Merge configuration sources with proper precedence:
        1. Environment variables (highest priority)
        2. Explicit keyword arguments
        3. Class defaults (lowest priority)
        """
        # CORRECTED MERGE ORDER
        return {**self.__dict__, **kwargs, **env_vars}  # Reversed env_vars/kwargs order

    @validator('cli_path', always=True)
    def validate_cli_path(cls, v):
        if not v:
            raise ValueError(
                "LLAMA_CLI_PATH must be set via environment variable "
                "or passed explicitly to the constructor"
            )
        if not os.path.exists(v):
            raise FileNotFoundError(f"llama-cli executable not found at {v}")
        if not os.access(v, os.X_OK):
            raise PermissionError(f"llama-cli at {v} is not executable")
        return v

    @validator('model_path', always=True)
    def validate_model_path(cls, v):
        if not v:
            raise ValueError(
                "MODEL_PATH must be set via environment variable "
                "or passed explicitly to the constructor"
            )
        if not os.path.exists(v):
            raise FileNotFoundError(f"Model file not found at {v}")
        return v

    @property
    def _llm_type(self) -> str:
        return "custom_llama_cli"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {field: getattr(self, field) for field in [
            'cli_path', 'model_path', 'n_ctx', 'n_threads', 'n_predict',
            'temperature', 'top_k', 'top_p', 'repeat_penalty', 'n_gpu_layers',
            'stop_sequences'
        ]}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        effective_stop = kwargs.get('stop', stop or self.stop_sequences)
        final_prompt = self._augment_prompt(prompt, effective_stop)
        
        try:
            output = self._execute_llama(final_prompt, effective_stop, **kwargs)
            return self._postprocess_output(output, prompt, final_prompt, effective_stop)
        except subprocess.TimeoutExpired:
            logger.warning("Generation timeout occurred")
            return "Generation timeout occurred"
        except subprocess.CalledProcessError as e:
            logger.error(f"Execution failed: {e.stderr.strip()}")
            raise ValueError(f"llama-cli error: {e.stderr.strip()}") from e

    def _augment_prompt(self, prompt: str, stop_sequences: List[str]) -> str:
        if stop_sequences:
            return f"{prompt}\n\nInstruction: Complete your response and end with one of these: {', '.join(stop_sequences)}"
        return prompt

    def _build_command(self, prompt: str, stop_sequences: List[str], **kwargs) -> List[str]:
        cmd = [
            self.cli_path,
            "-m", self.model_path,
            "-p", prompt,
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "--n-predict", str(kwargs.get('n_predict', self.n_predict)),
            "--temp", str(kwargs.get('temperature', self.temperature)),
            "--top-p", str(kwargs.get('top_p', self.top_p)),
            "--top-k", str(kwargs.get('top_k', self.top_k)),
            "--repeat-penalty", str(kwargs.get('repeat_penalty', self.repeat_penalty)),
            "--n-gpu-layers", str(self.n_gpu_layers)
        ]
        if stop_sequences:
            cmd += ["--stop"] + stop_sequences
        return cmd

    def _execute_llama(self, prompt: str, stop_sequences: List[str], **kwargs) -> str:
        cmd = self._build_command(prompt, stop_sequences, **kwargs)
        logger.debug(f"Executing: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=kwargs.get('timeout', 180)
        )
        if result.stderr:
            logger.debug(f"llama-cli stderr: {result.stderr.strip()}")
        return result.stdout.strip()

    def _postprocess_output(self, output: str, original_prompt: str, 
                          augmented_prompt: str, stop_sequences: List[str]) -> str:
        # Remove prompt from output
        for p in [augmented_prompt, original_prompt]:
            if output.startswith(p):
                output = output[len(p):].lstrip()
                break
        
        # Truncate at first stop sequence
        for seq in stop_sequences or []:
            if seq in output:
                output = output.split(seq)[0]
                break
                
        return output

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if os.environ.get("LLAMA_CLI_PATH") and os.environ.get("MODEL_PATH"):
        try:
            llm = CustomLlamaCliLLM()
            test_prompt = "Explain quantum computing in simple terms."
            print(f"Testing with prompt: {test_prompt}")
            print(llm(test_prompt))
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Set LLAMA_CLI_PATH and MODEL_PATH environment variables first")