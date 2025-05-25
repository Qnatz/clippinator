from __future__ import annotations

import os
import pickle
import logging
from pathlib import Path
from typing import Optional

from langchain.chains.llm import LLMChain
from langchain.agents import AgentExecutor, create_react_agent
from ..llms.llama_cli_llm import CustomLlamaCliLLM

from clippinator.project import Project
from clippinator.tools import get_tools, SimpleTool
from clippinator.tools.subagents import Subagent
from clippinator.tools.tool import WarningTool
from .base_minion import (
    CustomPromptTemplate,
    extract_variable_names, # We will still keep this but won't use it for explicit input variables
    BasicLLM,
)
from .executioner import Executioner, get_specialized_executioners
from .prompts import taskmaster_prompt, summarize_prompt, format_description, get_selfcall_objective
from ..tools.utils import ask_for_feedback
from ..utils.monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)

class DebuggingAgentExecutor(AgentExecutor):
    performance_monitor: Optional[PerformanceMonitor] = None

    def _call(self, inputs: dict, **kwargs) -> dict:
        try:
            result = super()._call(inputs, **kwargs)
            if self.performance_monitor:
                self.performance_monitor.log_execution(result, success=True)
            return result
        except Exception as e:
            if self.performance_monitor:
                self.performance_monitor.log_execution({}, success=False)
            raise e

class Taskmaster:
    def __init__(
        self,
        project: Project,
        prompt: CustomPromptTemplate | None = None,
        inner_taskmaster: bool = False
    ):
        self.project = project
        self.performance_monitor = PerformanceMonitor()
        self.specialized_executioners = get_specialized_executioners(project)
        self.default_executioner = Executioner(project)
        self.inner_taskmaster = inner_taskmaster
        
        if not hasattr(project, 'config'):
            project.config = {}
            
        try:
            self.llm = CustomLlamaCliLLM(
                cli_path=self.project.config.get('cli_path', '/default/path/to/llama-cli'),
                model_path=self.project.config.get('model_path', '/default/path/to/model.gguf'),
                n_ctx=self.project.config.get('n_ctx', 2048),
                n_threads=self.project.config.get('n_threads', 4)
            )
        except ValueError as e:
            logger.error(f"LLM initialization failed: {e}")
            raise RuntimeError("Failed to initialize LLM backend") from e

        self.tools = get_tools(project) # Store tools as an instance variable
        self.tools.append(SelfCall(project).get_tool(try_structured=False))

        agent_tool_names = [
            'DeclareArchitecture', 'ReadFile', 'WriteFile', 'Bash',
            'BashBackground', 'Human', 'Remember', 'TemplateInfo',
            'TemplateSetup', 'SetCI', 'Search'
        ]

        if not inner_taskmaster:
            agent_tool_names.append('SelfCall')

        self.tools.extend([ # Extend the instance variable
            Subagent(
                project, self.specialized_executioners, self.default_executioner
            ).get_tool(),
            WarningTool().get_tool()
        ])

        # Define explicit input variables for the prompt template
        # Ensure these cover all keys expected by your prompt's template string
        input_vars = [
            "objective", # This will be mapped to "input" for the agent
            "project_name",
            "project_summary", 
            "architecture",
            "history",
            "tools",         # Used by the prompt to list available tools
            "tool_names",    # Used by the prompt to list tool names
            "agent_scratchpad", # Crucial for ReAct agent's internal state
            "input"          # The primary input for the agent's reasoning
        ]
        
        # We need to ensure that 'base_prompt' is available, it was 'taskmaster_prompt' before.
        # Assuming `taskmaster_prompt` is your `base_prompt` for the agent.
        self.prompt = prompt or CustomPromptTemplate(
            template=taskmaster_prompt, # Changed from base_prompt to taskmaster_prompt
            tools=self.tools, # Use the instance variable
            input_variables=input_vars,  # Use explicit list
            agent_toolnames=agent_tool_names,
            my_summarize_agent=BasicLLM(
                base_prompt=summarize_prompt,
                #llm=self.llm # Corrected: Pass the LLM instance
            ),
            project=project,
            # Additional parameters like 'keep_n_last_thoughts', 'max_context_length' 
            # if they were previously implicitly set or needed.
            # You might need to adjust CustomPromptTemplate's __init__ to handle these.
            keep_n_last_thoughts=project.config.get('keep_n_last_thoughts', 2), # Example
            max_context_length=project.config.get('max_context_length', 10) # Example
        )
        self.prompt.hook = lambda _: self.save_to_file()

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools, # Use the instance variable
            prompt=self.prompt
        )

        self.agent_executor = DebuggingAgentExecutor(
            agent=agent,
            tools=self.tools, # Use the instance variable
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=project.config.get('max_iterations', 50),
            return_intermediate_steps=True,
            performance_monitor=self.performance_monitor
        )

    def run(self, inputs: dict) -> dict:
        # Merge static project data with dynamic agent requirements
        full_inputs = {
            **inputs, # Start with existing inputs (e.g., from project.prompt_fields())
            "tools": "\n".join([f"{t.name}: {t.description}" for t in self.tools]),
            "tool_names": [t.name for t in self.tools],
            # Initialize scratchpad. This will be updated by the agent's run cycle.
            # LangChain's ReAct agent internally manages the scratchpad,
            # but providing an initial empty one or ensuring the prompt can handle
            # its absence in the first call is good practice.
            "agent_scratchpad": "", 
            "input": inputs.get("objective", "")  # Map objective to input
        }
        
        # Ensure all required prompt input_variables are present in full_inputs
        # If any are missing from `inputs`, provide default empty strings
        for var in self.prompt.input_variables:
            if var not in full_inputs:
                full_inputs[var] = "" # Provide a default empty string for missing variables
                logger.warning(f"Missing expected input variable '{var}'. Setting to empty string.")

        try:
            # The agent_executor.invoke method already handles the prompt formatting
            # internally using its agent's prompt. We just need to pass the
            # dictionary of inputs it expects.
            result = self.agent_executor.invoke(full_inputs)
            self.project.update_state(result.get('output', ''))
            return result
        except KeyboardInterrupt:
            feedback = ask_for_feedback(lambda: self.project.menu(self.prompt))
            if feedback:
                # Append feedback to intermediate_steps, which will then be
                # picked up by the next call's agent_scratchpad formatting.
                self.prompt.intermediate_steps.append({"type": "human_feedback", "value": feedback})
            return self.run(inputs) # Use original inputs for re-run

    def save_to_file(self, path: str = ""):
        if not os.path.exists(self.project.path):
            return

        save_path = Path(path or self.project.path) / ".clippinator.pkl"
        save_data = {
            "prompt_data": {
                "current_context_length": self.prompt.current_context_length,
                "model_steps_processed": self.prompt.model_steps_processed,
                "all_steps_processed": self.prompt.all_steps_processed,
                "intermediate_steps": self.prompt.intermediate_steps,
                "last_summary": self.prompt.last_summary,
            },
            "project": self.project
        }

        with save_path.open("wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def load_from_file(cls, path: str | Path) -> Taskmaster:
        load_path = Path(path)
        with load_path.open("rb") as f:
            data = pickle.load(f)

        tm = cls(data["project"])
        tm.prompt.current_context_length = data["prompt_data"]["current_context_length"]
        tm.prompt.model_steps_processed = data["prompt_data"]["model_steps_processed"]
        tm.prompt.all_steps_processed = data["prompt_data"]["all_steps_processed"]
        tm.prompt.intermediate_steps = data["prompt_data"]["intermediate_steps"]
        tm.prompt.last_summary = data["prompt_data"]["last_summary"]
        return tm

class SelfCall(SimpleTool):
    name = "SelfCall"
    description = "Initializes project subcomponents. Use for each planned subfolder."

    def __init__(self, project: Project):
        super().__init__()
        self.initial_project = project

    def structured_func(self, sub_folder: str) -> str:
        sub_path = Path(self.initial_project.path) / sub_folder
        sub_path.mkdir(exist_ok=True)

        objective = get_selfcall_objective(
            self.initial_project.objective,
            self.initial_project.architecture,
            sub_folder
        )

        sub_project = Project(
            str(sub_path),
            objective,
            config=self.initial_project.config
        )

        taskmaster = Taskmaster(sub_project, inner_taskmaster=True)
        taskmaster.run(sub_project.prompt_fields())
        return f"Initialized subproject: {sub_folder}"

    def func(self, args: str) -> str:
        return self.structured_func(args.strip())

