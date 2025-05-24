from __future__ import annotations

import os
import pickle
import logging # Add logging import

from langchain.chains.llm import LLMChain # Updated import for consistency
from langchain.agents import AgentExecutor, create_react_agent # Updated imports
from ..llms.llama_cli_llm import CustomLlamaCliLLM # Add CustomLlamaCliLLM import

from clippinator.project import Project
from clippinator.tools import get_tools, SimpleTool
from clippinator.tools.subagents import Subagent
from clippinator.tools.tool import WarningTool
from .base_minion import (
    CustomPromptTemplate,
    extract_variable_names,
    # get_model, <--- REMOVE THIS
    BasicLLM,
)
from .executioner import Executioner, get_specialized_executioners
from .prompts import taskmaster_prompt, summarize_prompt, format_description, get_selfcall_objective
from ..tools.utils import ask_for_feedback

logger = logging.getLogger(__name__) # Initialize logger

class Taskmaster:
    def __init__(
            self,
            project: Project,
            prompt: CustomPromptTemplate | None = None,
            inner_taskmaster: bool = False
    ):
        self.project = project
        self.specialized_executioners = get_specialized_executioners(project)
        self.default_executioner = Executioner(project)
        self.inner_taskmaster = inner_taskmaster
        try:
            llm = CustomLlamaCliLLM()
        except ValueError as e:
            logger.error(f"Failed to initialize CustomLlamaCliLLM in Taskmaster: {e}")
            # Potentially raise a more specific error or handle it as per project's error handling strategy
            raise e
        tools = get_tools(project)
        tools.append(SelfCall(project).get_tool(try_structured=False))

        agent_tool_names = [
            'DeclareArchitecture', 'ReadFile', 'WriteFile', 'Bash', 'BashBackground', 'Human',
            'Remember', 'TemplateInfo', 'TemplateSetup', 'SetCI', 'Search'
        ]

        if not inner_taskmaster:
            agent_tool_names.append('SelfCall')

        tools.append(
            Subagent(
                project, self.specialized_executioners, self.default_executioner
            ).get_tool()
        )
        tools.append(WarningTool().get_tool())

        # ---- START DEBUG PRINT ----
        print(f"DEBUG: Taskmaster.__init__: About to create CustomPromptTemplate. Args:")
        print(f"DEBUG:   type(taskmaster_prompt): {type(taskmaster_prompt)}, taskmaster_prompt: '{str(taskmaster_prompt)[:100]}...'") # Print type and snippet
        print(f"DEBUG:   type(tools): {type(tools)}, len(tools): {len(tools) if tools is not None else 'None'}")
        _input_vars = extract_variable_names(taskmaster_prompt, interaction_enabled=True)
        print(f"DEBUG:   type(_input_vars): {type(_input_vars)}, _input_vars: {_input_vars}")
        print(f"DEBUG:   type(agent_tool_names): {type(agent_tool_names)}, agent_tool_names: {agent_tool_names}")
        # ---- END DEBUG PRINT ----

        self.prompt = prompt or CustomPromptTemplate(
            template=taskmaster_prompt,
            tools=tools,
            input_variables=_input_vars, # Use the pre-calculated _input_vars
            agent_toolnames=agent_tool_names,
            my_summarize_agent=BasicLLM(base_prompt=summarize_prompt), # This will trigger its own CustomLlamaCliLLM init
            project=project,
        )
        self.prompt.hook = lambda _: self.save_to_file()

        # llm_chain = LLMChain(llm=llm, prompt=self.prompt) # Removed
        # output_parser = CustomOutputParser() # CustomOutputParser is not directly used here

        agent = create_react_agent(
            llm=llm,
            tools=tools, # tools is the correct variable in Taskmaster
            prompt=self.prompt
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools, # tools is the correct variable in Taskmaster
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10, # As per instruction for debugging
            early_stopping_method="generate",
            return_intermediate_steps=True
        )

    def invoke(self, inputs: dict) -> dict: # Renamed run to invoke, input is dict
        # Ensure inputs is a mutable copy if we are modifying it.
        inputs = inputs.copy()

        inputs["specialized_minions"] = "\n".join(
            minion.expl() for minion in self.specialized_executioners.values()
        )
        inputs["format_description"] = format_description
        # Ensure 'input' is set for the agent if 'objective' is the primary input key.
        if 'objective' in inputs and 'input' not in inputs:
            inputs['input'] = inputs.pop('objective')
            
        try:
            result_dict = self.agent_executor.invoke(inputs) 
            # The problem description implies returning the dict, not just output string
            return result_dict 
            
        except KeyboardInterrupt:
            feedback = ask_for_feedback(lambda: self.project.menu(self.prompt))
            if feedback:
                self.prompt.intermediate_steps += [feedback] # Assuming feedback is compatible
            # Recursive call to self.invoke with original inputs (potentially modified by feedback mechanism)
            return self.invoke(inputs) # Changed to self.invoke

    def save_to_file(self, path: str = ""):
        if not os.path.exists(self.project.path):
            return
        path = path or os.path.join(self.project.path, f".clippinator.pkl")
        with open(path, "wb") as f:
            prompt = {
                "current_context_length": self.prompt.current_context_length,
                "model_steps_processed": self.prompt.model_steps_processed,
                "all_steps_processed": self.prompt.all_steps_processed,
                "intermediate_steps": self.prompt.intermediate_steps,
                "last_summary": self.prompt.last_summary,
            }
            pickle.dump((prompt, self.project), f)

    @classmethod
    def load_from_file(cls, path: str = ".clippinator.pkl"):
        with open(path, "rb") as f:
            prompt, project = pickle.load(f)
        self = cls(project)
        self.prompt.current_context_length = prompt["current_context_length"]
        self.prompt.model_steps_processed = prompt["model_steps_processed"]
        self.prompt.all_steps_processed = prompt["all_steps_processed"]
        self.prompt.intermediate_steps = prompt["intermediate_steps"]
        self.prompt.last_summary = prompt["last_summary"]
        return self


class SelfCall(SimpleTool):
    name = "SelfCall"
    description = "Initializes the component of the project. " \
                  "It's highly advised to use this tool for each subfolder from the " \
                  "\"planned project architecture\" by Architect when this subfolder does not exist in the " \
                  "current state of project (all folders and files) (or the project structure is empty). " \
                  "It's A MUST to use this tool right after the Subagent @Architect for every subfolder " \
                  "from the \"planned project architecture\"." \
                  "Input parameter - name of the subfolder, a relative path to subfolder from the current location."

    def __init__(self, project: Project):
        self.initial_project = project
        super().__init__()

    def structured_func(self, sub_folder: str):
        sub_project_path = self.initial_project.path + (
            "/" if not self.initial_project.path.endswith("/") else "") + sub_folder
        cur_objective = self._get_resulting_objective(self.initial_project, sub_folder)
        cur_sub_project = Project(sub_project_path, cur_objective, architecture="")
        taskmaster = Taskmaster(cur_sub_project, inner_taskmaster=True)
        # Call the public run method of the Taskmaster instance
        taskmaster_inputs = cur_sub_project.prompt_fields()
        # Ensure 'input' key if the nested Taskmaster's agent expects it
        if 'objective' in taskmaster_inputs and 'input' not in taskmaster_inputs:
             taskmaster_inputs['input'] = taskmaster_inputs.pop('objective')
        taskmaster.invoke(taskmaster_inputs) # Changed to taskmaster.invoke
        return f"{sub_folder} folder processed."

    def func(self, args: str):
        sub_folder = args.strip()
        return self.structured_func(sub_folder)

    @staticmethod
    def _get_resulting_objective(initial_project: Project, sub_folder: str) -> str:
        return get_selfcall_objective(
            initial_project.objective,
            initial_project.architecture,
            sub_folder
        )
