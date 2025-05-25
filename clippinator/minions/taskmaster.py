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
    extract_variable_names,
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
                cli_path=project.config.get('cli_path'),
                model_path=project.config.get('model_path'),
                n_ctx=project.config.get('n_ctx', 2048),
                n_threads=project.config.get('n_threads', 4)
            )
        except ValueError as e:
            logger.error(f"LLM initialization failed: {e}")
            raise RuntimeError("Failed to initialize LLM backend") from e

        tools = get_tools(project)
        tools.append(SelfCall(project).get_tool(try_structured=False))

        agent_tool_names = [
            'DeclareArchitecture', 'ReadFile', 'WriteFile', 'Bash',
            'BashBackground', 'Human', 'Remember', 'TemplateInfo',
            'TemplateSetup', 'SetCI', 'Search'
        ]

        if not inner_taskmaster:
            agent_tool_names.append('SelfCall')

        tools.extend([
            Subagent(
                project, self.specialized_executioners, self.default_executioner
            ).get_tool(),
            WarningTool().get_tool()
        ])

        _input_vars = extract_variable_names(taskmaster_prompt, interaction_enabled=True)

        self.prompt = prompt or CustomPromptTemplate(
            template=taskmaster_prompt,
            tools=tools,
            input_variables=_input_vars,
            agent_toolnames=agent_tool_names,
            my_summarize_agent=BasicLLM(
                base_prompt=summarize_prompt,
                llm=self.llm
            ),
            project=project,
        )
        self.prompt.hook = lambda _: self.save_to_file()

        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=self.prompt
        )

        self.agent_executor = DebuggingAgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=project.config.get('max_iterations', 50),
            return_intermediate_steps=True,
            performance_monitor=self.performance_monitor
        )

    def run(self, inputs: dict) -> dict:
        processed_inputs = inputs.copy()

        # Handle objective/input key conversion
        if 'objective' in processed_inputs:
            processed_inputs['input'] = processed_inputs.pop('objective')

        try:
            # Format inputs with proper scratchpad
            formatted_inputs = self.prompt.format(
                **processed_inputs,
                agent_scratchpad=self.prompt.format_scratchpad(
                    self.prompt.intermediate_steps
                )
            )
            result = self.agent_executor.invoke(formatted_inputs)
            self.project.update_state(result.get('output', ''))
            return result
        except KeyboardInterrupt:
            feedback = ask_for_feedback(lambda: self.project.menu(self.prompt))
            if feedback:
                self.prompt.intermediate_steps.append(feedback)
            return self.run(processed_inputs)

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