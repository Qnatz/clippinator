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

# Corrected import path assuming taskmaster.py is in clippinator/minions/
# and monitoring.py is in clippinator/utils/
from ..utils.monitoring import PerformanceMonitor 
from typing import Optional # Import Optional for type hinting


class DebuggingAgentExecutor(AgentExecutor):
    performance_monitor: Optional[PerformanceMonitor] = None # Declare as a class variable

    # The __init__ method can remain as is, or be removed if no custom logic beyond
    # what AgentExecutor's __init__ does is needed. Pydantic will handle passing
    # 'performance_monitor' to this field if it's provided in the constructor.
    # For clarity and explicit intent, keeping the __init__ but ensuring it calls super
    # correctly and handles the performance_monitor if passed.
    # However, the error `ValueError: "DebuggingAgentExecutor" object has no field "performance_monitor"`
    # suggests that Pydantic v1 (which AgentExecutor uses) relies on class variable annotations
    # for field definition, and then __init__ (often the base class's) populates them.
    # So, the primary fix is the class variable.
    # Let's simplify the __init__ or ensure it correctly passes kwargs to super if we keep it.

    # Option 1: Simplified __init__ (relying on Pydantic to populate the declared field)
    # No explicit __init__ needed unless we have other custom initialization logic.
    # Pydantic will automatically use the performance_monitor kwarg if provided.

    # Option 2: Keep __init__ but ensure it's Pydantic-friendly
    # If we keep __init__, we must ensure it doesn't conflict with Pydantic's field handling.
    # The original __init__ was:
    # def __init__(self, *args, performance_monitor: PerformanceMonitor = None, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.performance_monitor = performance_monitor # This line is problematic if performance_monitor is not a recognized field by Pydantic.
    # By declaring it as a class variable, it becomes a recognized field.
    # The assignment `self.performance_monitor = performance_monitor` within __init__
    # might be redundant if Pydantic handles it, or necessary if we want to allow it
    # to be set both via kwarg at instantiation and potentially later.
    # Given Pydantic v1 behavior, just declaring the field and letting the base __init__
    # (called via super) handle kwargs is usually sufficient.

    # Let's stick to just declaring the class variable, as that's the core Pydantic way.
    # The AgentExecutor's __init__ will take care of populating it if 'performance_monitor'
    # is in the kwargs.

    def _call(self, inputs: dict, **kwargs) -> dict: # Ensure type hint for inputs
        print(f"\n=== NEW EXECUTION CYCLE ===")
        # Ensure 'input' key exists before trying to print it, or print all inputs
        # The original example was: print(f"Input: {inputs}")
        # Let's print the whole inputs dictionary for better debugging.
        print(f"Inputs: {inputs}")
        
        success_flag = True # Assume success unless an exception occurs
        result = {} # Initialize result to an empty dict to ensure it's always defined
        try:
            result = super()._call(inputs, **kwargs)
        except Exception as e:
            # If super()._call raises an exception, it's an error in execution
            success_flag = False
            # Log the error if needed, then re-raise or handle
            if self.performance_monitor:
                # Pass empty dict for result as it might not be fully formed.
                self.performance_monitor.log_execution({}, success=False) 
            raise e # Re-raise the exception

        # Log execution to PerformanceMonitor if available
        if self.performance_monitor:
            # Determine success based on whether an error occurred or if 'output' indicates failure
            # For now, using the success_flag from the try-except block.
            # A more sophisticated check might involve inspecting result['output'].
            self.performance_monitor.log_execution(result, success=success_flag)

        print("\n=== INTERMEDIATE STEPS ===")
        # Ensure 'intermediate_steps' is in result and is iterable
        if 'intermediate_steps' in result and result['intermediate_steps'] is not None:
            for i, step in enumerate(result['intermediate_steps']):
                action, observation = step
                # Check if action is an AgentAction object and has tool and tool_input attributes
                if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                    print(f"Step {i+1}:")
                    print(f"Action: {action.tool} | Input: {action.tool_input}")
                    print(f"Observation: {str(observation)[:200]}...") # str() conversion for safety
                else:
                    print(f"Step {i+1}: Action/Observation format not as expected: {step}")
        else:
            # This handles cases where result might be empty due to an early exception
            # or if 'intermediate_steps' is genuinely not in the result.
            print("No intermediate steps found or result['intermediate_steps'] is None.")


        print("\n=== FINAL OUTPUT ===")
        # Ensure 'output' key exists in result
        if 'output' in result:
            print(result['output'])
        else:
            # This handles cases where result might be empty or 'output' is missing.
            print("No 'output' key found in result.")
        
        return result


class Taskmaster:
    def __init__(
            self,
            project: Project,
            prompt: CustomPromptTemplate | None = None,
            inner_taskmaster: bool = False
    ):
        self.project = project
        self.performance_monitor = PerformanceMonitor() # Instantiate the monitor
        self.specialized_executioners = get_specialized_executioners(project)
        self.default_executioner = Executioner(project)
        self.inner_taskmaster = inner_taskmaster
        try:
            llm = CustomLlamaCliLLM(
                n_ctx=4096,
                n_predict=512,
                temperature=0.7,
                top_k=50,
                top_p=0.85,
                repeat_penalty=1.15,
                stop_sequences=["\nFinal Answer:", "<|im_end|>"]
            )
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
        # self.agent_executor = AgentExecutor(
        self.agent_executor = DebuggingAgentExecutor( # Use the new class
            agent=agent,
            tools=tools, # tools is already defined and used by create_react_agent
            verbose=True, # Retain verbose, or let DebuggingAgentExecutor handle all prints
            handle_parsing_errors=True, # Retain
            max_iterations=8, # Retain
            # early_stopping_method="generate", # Removed to avoid ValueError
            return_intermediate_steps=True, # Retain, as DebuggingAgentExecutor uses it
            performance_monitor=self.performance_monitor # Pass the instance
        )

    def generate_performance_report(self):
        if hasattr(self, 'performance_monitor') and self.performance_monitor is not None:
            self.performance_monitor.generate_report()
        else:
            print("Performance monitor not available.")

    def invoke(self, inputs: dict) -> dict: # Renamed run to invoke, input is dict
        # Ensure inputs is a mutable copy if we are modifying it.
        inputs = inputs.copy()

        # The new prompt uses {objective} and {history}
        # {objective} is already in inputs (usually)
        # {history} maps to agent_scratchpad / intermediate_steps
        # {tools} is handled by the CustomPromptTemplate
        # {project_name} is also in project.prompt_fields()

        # We need to ensure the 'history' key is populated for the new prompt.
        # The CustomPromptTemplate populates 'agent_scratchpad' which is used by ReAct.
        # For the new prompt, we'll explicitly add 'history' if not present,
        # potentially duplicating what format_messages does, but ensuring it's there.
        if 'history' not in inputs:
            inputs['history'] = self.prompt.construct_scratchpad(self.prompt.intermediate_steps)


        # specialized_minions and format_description were for the old prompt.
        # Remove them if they are not expected by the new prompt structure.
        # inputs["specialized_minions"] = "\n".join(
        #     minion.expl() for minion in self.specialized_executioners.values()
        # )
        # inputs["format_description"] = format_description

        # Ensure 'input' is set for the agent if 'objective' is the primary input key.
        # The new prompt uses {objective}, so this mapping might be less critical
        # if 'objective' is consistently used. But keeping it for now.
        if 'objective' in inputs and 'input' not in inputs: # 'input' is a common key for Langchain agents
            inputs['input'] = inputs['objective']


        try:
            # The agent_executor.invoke expects 'input' for the main query.
            # Other variables like 'objective', 'project_name', 'tools', 'history'
            # should be correctly formatted into the prompt by CustomPromptTemplate.
            # Ensure the primary input to the agent is under the key 'input'.
            # If the main objective is in 'objective', make sure it's also in 'input'.
            
            # The CustomPromptTemplate will take care of formatting the prompt
            # with all necessary variables including 'objective', 'project_name', 'tools', 'history'.
            # The 'input' variable for agent_executor.invoke is the primary query or task.
            # In our case, this is the project's objective.
            
            current_objective = inputs.get('input', inputs.get('objective', ''))
            if not current_objective:
                logger.error("Objective is missing in inputs for Taskmaster.invoke")
                raise ValueError("Objective must be provided in inputs.")

            # All other necessary variables like project_name, tools, history (agent_scratchpad)
            # are formatted by the CustomPromptTemplate.
            # The key for the main input to the agent executor is 'input'.
            agent_inputs = {'input': current_objective, **inputs}


            result_dict = self.agent_executor.invoke(agent_inputs)
            return result_dict

        except KeyboardInterrupt:
            feedback = ask_for_feedback(lambda: self.project.menu(self.prompt))
            if feedback:
                # Assuming feedback is a list of (action, observation) tuples
                # or a string that needs to be wrapped in appropriate structure
                # This part might need adjustment based on actual feedback format
                if isinstance(feedback, str): # Basic handling if feedback is just a string
                    self.prompt.intermediate_steps.append(("\nSystem note (feedback):", feedback))
                elif isinstance(feedback, list): # If feedback is already structured
                    self.prompt.intermediate_steps.extend(feedback)
                else: # Fallback or more specific handling
                    logger.warning(f"Received feedback of unexpected type: {type(feedback)}")

            return self.invoke(inputs) # Recursive call with original inputs

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
