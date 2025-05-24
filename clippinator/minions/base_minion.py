from __future__ import annotations

import os
import re
import logging # Added import
from dataclasses import dataclass, field # Updated import
from typing import List, Union, Callable, Any, Optional # Added Optional

import langchain.schema
from langchain.chains.llm import LLMChain # Updated import
from langchain_core.prompts import PromptTemplate # Updated import
from langchain.agents import (
    Tool,
    AgentExecutor,
    create_react_agent, # Ensured this is present
    # AgentOutputParser, # Removed as CustomOutputParser is being removed
)
# Removed OpenAIFunctionsAgent, ChatOpenAI, ChatAnthropic
# from langchain_community.llms import LlamaCpp # Removed LlamaCpp import
from ..llms.llama_cli_llm import CustomLlamaCliLLM # Added CustomLlamaCliLLM import
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish

from clippinator.tools.tool import WarningTool
from clippinator.project.project import Project # Added Project import
from .prompts import format_description
from ..tools.utils import trim_extra, ask_for_feedback

logger = logging.getLogger(__name__) # Added logger

long_warning = (
    "WARNING: You have been working for a very long time. Please, finish ASAP. "
    "If there are obstacles, please, return with the result and explain the situation."
)


def remove_surrogates(text):
    return "".join(c for c in text if not ('\ud800' <= c <= '\udfff'))


# CustomOutputParser class definition removed.

def remove_project_summaries(text: str) -> str:
    """
    Remove all the project summaries from the text EXCEPT for the last occurrence
    The project summary is between "Current project state:" and "---"
    """
    # Find all the project summaries
    project_summaries = re.findall(r"Current project state:.*?-----", text, re.DOTALL)
    # Remove all the project summaries except for the last one
    for project_summary in project_summaries[:-1]:
        text = text.replace(project_summary, "", 1)
    return text


def extract_variable_names(prompt: str, interaction_enabled: bool = False):
    variable_pattern = r"\{(\w+)\}"
    variable_names = re.findall(variable_pattern, prompt)
    if interaction_enabled:
        for name in ["tools", "tool_names", "agent_scratchpad"]:
            if name in variable_names:
                variable_names.remove(name)
        variable_names.append("intermediate_steps")
    return variable_names

# Removed get_model function

@dataclass
class BasicLLM:
    base_prompt: str  # Field to be provided in __init__
    
    # Fields to be initialized in __post_init__
    prompt: PromptTemplate = field(init=False)
    llm: LLMChain = field(init=False)

    def __post_init__(self):
        try:
            llm_instance = CustomLlamaCliLLM() 
        except ValueError as e:
            logger.error(f"Failed to initialize CustomLlamaCliLLM in BasicLLM: {e}")
            raise e 
        
        self.prompt = PromptTemplate(
            template=self.base_prompt, # Use self.base_prompt here
            input_variables=extract_variable_names(self.base_prompt),
        )
        self.llm = LLMChain( 
            llm=llm_instance,
            prompt=self.prompt, 
        )

    def invoke(self, inputs: dict) -> str:
        inputs["feedback"] = inputs.get("feedback", "") # Ensure 'feedback' is handled if necessary
        # self.llm is an LLMChain
        response = self.llm.invoke(inputs)
        # Extract text, default to empty string if 'text' key not found or response is not dict
        return response.get('text', '') if isinstance(response, dict) else str(response)


class CustomPromptTemplate(StringPromptTemplate):
    template: str
    # The list of tools available
    tools: List[Tool]
    agent_toolnames: List[str]
    max_context_length: int
    keep_n_last_thoughts: int
    project: Optional['Project'] 
    my_summarize_agent: Optional['BasicLLM'] 
    hook: Optional[Callable[['CustomPromptTemplate'], None]]
    # current_context_length: int = 5 # Now an instance variable
    # keep_n_last_thoughts: int = 2 # Now an instance variable
    current_context_length: int = 0
    model_steps_processed: int = 0
    all_steps_processed: int = 0
    # my_summarize_agent: Any = None # To be set as instance attribute in __init__
    last_summary: str = ""
    # project: Any | None = None # To be set as instance attribute in __init__
    intermediate_steps: list = field(default_factory=list) # Ensured default_factory
    # hook: Optional[Callable[[CustomPromptTemplate], None]] = None # To be set as instance attribute in __init__

    # Pydantic fields: template, tools, agent_toolnames (and input_variables from parent)
    # Other attributes: max_context_length, keep_n_last_thoughts, project, my_summarize_agent, hook (set in __init__)
    # State attributes with class defaults: current_context_length, model_steps_processed, all_steps_processed, last_summary

    def __init__(
        self,
        template: str,
        tools: List[Tool],
        agent_toolnames: List[str],
        input_variables: List[str],
        max_context_length: int = 5,
        keep_n_last_thoughts: int = 2,
        project: Optional['Project'] = None,
        my_summarize_agent: Optional['BasicLLM'] = None,
        hook: Optional[Callable[['CustomPromptTemplate'], None]] = None,
        **kwargs: Any
    ):
        super_kwargs = {
            "input_variables": input_variables,
            "template": template,
            "tools": tools,
            "agent_toolnames": agent_toolnames,
            "max_context_length": max_context_length,
            "keep_n_last_thoughts": keep_n_last_thoughts,
            "project": project,
            "my_summarize_agent": my_summarize_agent,
            "hook": hook,
        }
        
        for key in ["max_context_length", "keep_n_last_thoughts", "project", "my_summarize_agent", "hook"]:
            kwargs.pop(key, None) 
        
        super_kwargs.update(kwargs)
        super().__init__(**super_kwargs)
        
        # self.intermediate_steps: list = [] # Removed, handled by default_factory

    @property
    def _prompt_type(self) -> str:
        return "taskmaster"

    def thought_log(self, thoughts: list[tuple[AgentAction, str]]) -> str: # Corrected type hint
        result = ""
        # For ReAct, the 'thoughts' are the intermediate_steps (AgentAction, str_observation)
        for action, observation in thoughts:
            # action.log already contains "Thought: ...
Action: ...
Action Input: ..."
            result += action.log 
            # Append the observation, which is the result of the action.
            result += f"\nObservation: {str(observation)}\n" 
        return result

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, AResult tuples)
        # Format them in a particular way
        if 'intermediate_steps' in kwargs:
            model_steps = kwargs.pop("intermediate_steps")
            self.intermediate_steps += model_steps[self.model_steps_processed:]
            self.model_steps_processed = len(model_steps)
            intermediate_steps = self.intermediate_steps

            self.current_context_length += (
                    len(intermediate_steps) - self.all_steps_processed
            )
            self.all_steps_processed = len(intermediate_steps)

            if (
                    self.current_context_length >= self.max_context_length
                    and self.my_summarize_agent
            ):
                print(f"[INFO] Summarization triggered. Last summary length: {len(self.last_summary)}, current_context_length: {self.current_context_length}, max_context_length: {self.max_context_length}")
                summarizer_input = {
                    "summary": self.last_summary,
                    "thought_process": self.thought_log(
                        intermediate_steps[
                        -self.current_context_length: -self.keep_n_last_thoughts
                        ]
                    )
                }
                self.last_summary = self.my_summarize_agent.invoke(summarizer_input)
                self.current_context_length = self.keep_n_last_thoughts

            if self.my_summarize_agent:
                kwargs["agent_scratchpad"] = (
                        "Here is a summary of what has happened:\n" + trim_extra(self.last_summary, 2700, 1900)
                )
                kwargs["agent_scratchpad"] += "\nEND OF SUMMARY\n"
            else:
                kwargs["agent_scratchpad"] = ""

            kwargs["agent_scratchpad"] += "Here go your thoughts and actions:\n" # This part is fine

            # The 'intermediate_steps' for the current context length are formatted using the updated thought_log.
            current_thought_steps = intermediate_steps[-self.current_context_length:]
            kwargs["agent_scratchpad"] += self.thought_log(current_thought_steps)
            
        # Ensure 'input' is present in kwargs for self.template.format, 
        # if not already passed in from the Minion's invoke method.
        # This is a fallback, ideally 'input' is explicitly in kwargs.
        if 'input' not in kwargs:
            # 'objective' or 'task' might be the original key for the main input.
            # This depends on how BaseMinion/Taskmaster.invoke structures its 'inputs' dict.
            # Assuming 'objective' is a possible key for the main input if 'input' is missing.
            kwargs['input'] = kwargs.get('objective', kwargs.get('task', ''))


        kwargs["tools"] = "\n".join(
            [
                f"{tool.name}: {tool.description}"
                for tool in self.tools
                if tool.name in self.agent_toolnames
            ]
        )
        kwargs["tool_names"] = self.agent_toolnames
        if self.project:
            for key, value in self.project.prompt_fields().items():
                kwargs[key] = value
        # print("Prompt:\n\n" + self.template.format(**kwargs) + "\n\n\n")
        result = remove_surrogates(
            remove_project_summaries(self.template.format(**kwargs))) # Removed .replace('{tools}', kwargs['tools']) as format should handle it
        result = trim_extra(result, 25000)
        if self.hook:
            self.hook(self)
        if self.project and os.path.exists(self.project.path):
            with open(os.path.join(self.project.path, ".prompts.log"), "a") as f:
                f.write(result + "\n\n============================\n\n\n")
        return result


# This function seems to be duplicated, removing one instance.
# def extract_variable_names(prompt: str, interaction_enabled: bool = False):
#     variable_pattern = r"\{(\w+)\}"
#     variable_names = re.findall(variable_pattern, prompt)
#     if interaction_enabled:
#         for name in ["tools", "tool_names", "agent_scratchpad"]:
#             if name in variable_names:
#                 variable_names.remove(name)
#         variable_names.append("intermediate_steps")
#     return variable_names


@dataclass
class BaseMinion:
    def __init__(
            self,
            base_prompt,
            available_tools,
            max_iterations: int = 50, # This parameter is now overridden by a fixed value in AgentExecutor
            allow_feedback: bool = False,
            max_context_length: int = 5,
            keep_n_last_thoughts: int = 2
            # Removed LlamaCpp specific parameters
    ) -> None:
        try:
            llm = CustomLlamaCliLLM()
        except ValueError as e:
            logger.error(f"Failed to initialize CustomLlamaCliLLM in BaseMinion: {e}")
            raise e # Re-raise the exception after logging
        agent_toolnames = [tool.name for tool in available_tools]
        extended_tools = list(available_tools) # extended_tools is used for CustomPromptTemplate
        extended_tools.append(WarningTool().get_tool())

        self.prompt = CustomPromptTemplate(
            template=base_prompt,
            tools=extended_tools, 
            input_variables=extract_variable_names(
                base_prompt, interaction_enabled=True
            ),
            agent_toolnames=agent_toolnames,
            max_context_length=max_context_length,
            keep_n_last_thoughts=keep_n_last_thoughts,
        )

        # llm_chain = LLMChain(llm=llm, prompt=self.prompt) # Removed
        # output_parser is not directly used by create_react_agent, 
        # but CustomOutputParser might be part of ReAct agent's internal workings if not overridden.
        # For now, keep the variable definition if it's used elsewhere or if ReAct might use it.
        # If it's confirmed ReAct uses its own parser exclusively and CustomOutputParser is not needed by the agent,
        # then this line can be removed.
        # _output_parser = CustomOutputParser() # Instantiation removed.

        agent = create_react_agent(
            llm=llm,
            tools=available_tools, # create_react_agent uses `available_tools`
            prompt=self.prompt
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=available_tools, # AgentExecutor also uses `available_tools`
            verbose=True,
            handle_parsing_errors=True, 
            max_iterations=10, # Updated from max_iterations variable as per instructions
            early_stopping_method="generate", 
            return_intermediate_steps=True    
        )
        self.allow_feedback = allow_feedback

    def invoke(self, inputs: dict) -> dict:
        inputs["feedback"] = inputs.get("feedback", "")
        inputs["format_description"] = format_description # This was part of the original **kwargs
        
        # Simplest change for now, focusing on .invoke():
        return self.agent_executor.invoke(inputs)

# Removed BaseMinionOpenAI class

@dataclass
class FeedbackMinion:
    underlying_minion: BaseMinion | BasicLLM
    eval_llm: LLMChain
    feedback_prompt: str
    check_function: Callable[[str], Any]

    def __init__(
            self,
            minion: BaseMinion | BasicLLM,
            eval_prompt: str,
            feedback_prompt: str,
            check_function: Callable[[str], Any] = lambda x: None
            # Removed LlamaCpp specific parameters for the evaluation LLM
    ) -> None:
        try:
            eval_llm_instance = CustomLlamaCliLLM()
        except ValueError as e:
            logger.error(f"Failed to initialize CustomLlamaCliLLM in FeedbackMinion for eval_llm: {e}")
            raise e # Re-raise the exception after logging
        self.eval_llm = LLMChain(
            llm=eval_llm_instance,
            prompt=PromptTemplate(
                template=eval_prompt,
                input_variables=extract_variable_names(eval_prompt),
            ),
        )
        self.underlying_minion = minion
        self.feedback_prompt = feedback_prompt
        self.check_function = check_function

    def invoke(self, inputs: dict) -> dict:
        original_inputs = inputs.copy()

        if "feedback" in original_inputs and "previous_result" in original_inputs:
            print("Rerunning a prompt with feedback:", original_inputs["feedback"])
            if len(original_inputs["previous_result"]) > 500:
                original_inputs["previous_result"] = (
                        original_inputs["previous_result"][:500] + "\n...(truncated)\n"
                )
            # Ensure feedback_prompt is formatted correctly, using only keys present in original_inputs
            # This might require careful construction of feedback_prompt string template
            try:
                current_feedback_text = original_inputs.get("feedback", "")
                previous_result_text = original_inputs.get("previous_result", "")
                # Construct a dictionary with only the keys expected by feedback_prompt
                feedback_format_args = {
                    "feedback": current_feedback_text,
                    "previous_result": previous_result_text
                }
                # Add other relevant keys from original_inputs if feedback_prompt expects them
                for key in original_inputs:
                    if key not in feedback_format_args:
                        feedback_format_args[key] = original_inputs[key]
                
                original_inputs["feedback"] = self.feedback_prompt.format(**feedback_format_args)
            except KeyError as e:
                logger.warning(f"KeyError during feedback_prompt formatting: {e}. Using simpler format for feedback string.")
                original_inputs["feedback"] = f"Critique: {original_inputs.get('feedback', '')}\nOriginal work: {original_inputs.get('previous_result', '')}"

        underlying_response = self.underlying_minion.invoke(original_inputs)
        
        if isinstance(underlying_response, dict):
            res_output = underlying_response.get('output', '')
            final_return_value = underlying_response 
        else: 
            res_output = str(underlying_response)
            final_return_value = {"output": res_output} 
        
        check_error_msg = None
        try:
            self.check_function(res_output) 
        except ValueError as e:
            check_error_msg = " ".join(e.args)
        
        if check_error_msg: 
            new_inputs_for_retry = original_inputs.copy()
            new_inputs_for_retry["feedback"] = check_error_msg
            new_inputs_for_retry["previous_result"] = res_output
            return self.invoke(new_inputs_for_retry)
        
        eval_kwargs = {"result": res_output, **original_inputs}
        # Filter eval_kwargs to only include keys expected by the eval_llm's prompt
        expected_eval_vars = self.eval_llm.prompt.input_variables
        filtered_eval_kwargs = {k: v for k, v in eval_kwargs.items() if k in expected_eval_vars}
        
        evaluation_result = self.eval_llm.invoke(filtered_eval_kwargs)
        evaluation_text = evaluation_result.get('text', '') if isinstance(evaluation_result, dict) else str(evaluation_result)

        if "ACCEPT" in evaluation_text:
            return final_return_value
        
        new_inputs_for_retry_with_eval = original_inputs.copy()
        new_inputs_for_retry_with_eval["feedback"] = evaluation_text.split("Feedback: ", 1)[-1].strip()
        new_inputs_for_retry_with_eval["previous_result"] = res_output
        return self.invoke(new_inputs_for_retry_with_eval)
