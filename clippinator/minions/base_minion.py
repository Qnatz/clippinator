from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Union, Callable, Any, Optional
from pydantic.v1 import Field

import langchain.schema
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain.agents import (
    Tool,
    AgentExecutor,
    create_react_agent,
)
from ..llms.llama_cli_llm import CustomLlamaCliLLM
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish

from clippinator.tools.tool import WarningTool
from clippinator.project.project import Project
from .prompts import format_description
from ..tools.utils import trim_extra, ask_for_feedback

logger = logging.getLogger(__name__)

def custom_handle_parsing_errors(error: OutputParserException) -> str:
    """Custom handler for Langchain OutputParserExceptions."""
    error_message = str(error)
    # older versions of langchain have llm_output, newer have observation
    llm_output = getattr(error, 'llm_output', getattr(error, 'observation', 'No output available'))
    
    logger.error(f"Output parsing error: {error_message}")
    logger.error(f"Faulty LLM output: {llm_output}")

    # Ensure llm_output is a string and truncate if too long
    if not isinstance(llm_output, str):
        llm_output = str(llm_output)
    if len(llm_output) > 1000: # Truncate for prompt
        llm_output = llm_output[:1000] + "..."

    return (
        "Your previous response was not formatted correctly.\n"
        f"Error: {error_message}\n"
        f"Faulty Output: {llm_output}\n"
        "You MUST use the following format:\n"
        "Thought: Your reasoning and thought process.\n"
        "Action: The action to take.\n"
        "Action Input: The input to the action.\n"
        "Observation: The result of the action.\n"
        "If you are stuck or unsure what to do, try using a general-purpose tool like 'Human' or a specific 'ErrorRecoveryTool' if available.\n"
        "Ensure 'Action:' and 'Action Input:' are present and correctly formatted.\n"
    )

long_warning = (
    "WARNING: You have been working for a very long time. Please, finish ASAP. "
    "If there are obstacles, please, return with the result and explain the situation."
)


def run_with_retries(agent_executor, inputs, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = agent_executor.invoke(inputs)
            return result
        except Exception as e:
            if "Invalid Format" in str(e):
                print(f"Format error on attempt {attempt + 1}: {e}")
                # Add format reminder to inputs
                inputs['agent_scratchpad'] = inputs.get('agent_scratchpad', '') + "\n\nREMINDER: You MUST follow the Thought -> Action -> Action Input format. Every Thought needs an Action!"
            elif "timeout" in str(e).lower():
                print(f"Timeout on attempt {attempt + 1}")
                # Reduce complexity or split task (Note: actual task splitting is complex and might be out of scope for this function, print is a placeholder)
            else:
                print(f"Unexpected error: {e}")
            
            if attempt == max_retries - 1:
                raise
    return None # Should not be reached if max_retries > 0 due to raise


def remove_surrogates(text):
    return "".join(c for c in text if not ('\ud800' <= c <= '\udfff'))


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


def extract_variable_names(prompt: str, interaction_enabled: bool = False) -> List[str]:
    """Modified to ensure agent requirements are always met"""
    variable_pattern = r"\{(\w+)\}"
    variables = set(re.findall(variable_pattern, prompt))
    
    if interaction_enabled:
        # Always include required agent variables
        required = {"tools", "tool_names", "agent_scratchpad", "input"}
        variables.update(required)
    
    return list(variables)


@dataclass
class BasicLLM:
    base_prompt: str
    
    prompt: PromptTemplate = field(init=False)
    llm: LLMChain = field(init=False)

    def __post_init__(self):
        try:
            llm_instance = CustomLlamaCliLLM() 
        except ValueError as e:
            logger.error(f"Failed to initialize CustomLlamaCliLLM in BasicLLM: {e}")
            raise e 
        
        self.prompt = PromptTemplate(
            template=self.base_prompt,
            input_variables=extract_variable_names(self.base_prompt),
        )
        self.llm = LLMChain( 
            llm=llm_instance,
            prompt=self.prompt, 
        )

    def invoke(self, inputs: dict) -> str:
        inputs["feedback"] = inputs.get("feedback", "")
        # The prompt_fields from project will add 'history', so no need for a fallback here if project is always available
        # inputs["history"] = ""  
        # Ensure project fields are merged if self.project is available and needs to update inputs
        # If BasicLLM is used independently of Project context, this needs to be optional or handled
        # based on where it's called (e.g., from CustomPromptTemplate where project is known).
        # For now, assuming it's called in a context where 'history' and other fields are already in 'inputs'
        # or it is intended to operate on a simpler prompt.
        # This line was problematic as BasicLLM doesn't have self.project unless explicitly passed.
        # inputs.update(self.project.prompt_fields()) # Removed as BasicLLM doesn't have self.project
        
        response = self.llm.invoke(inputs)
        return response.get('text', '') if isinstance(response, dict) else str(response)


class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    agent_toolnames: List[str]
    max_context_length: int
    keep_n_last_thoughts: int
    project: Optional['Project'] 
    my_summarize_agent: Optional['BasicLLM'] 
    hook: Optional[Callable[['CustomPromptTemplate'], None]]
    current_context_length: int = 0
    model_steps_processed: int = 0
    all_steps_processed: int = 0
    last_summary: str = ""
    intermediate_steps: list = Field(default_factory=list)

    def __init__(
        self,
        template: str,
        tools: List[Tool],
        agent_toolnames: List[str],
        input_variables: List[str],
        max_context_length: int = 15,
        keep_n_last_thoughts: int = 10,
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
        

    @property
    def _prompt_type(self) -> str:
        return "taskmaster"

    def format_scratchpad(self, intermediate_steps: list[tuple[AgentAction, str]]) -> str:
        """Formats the agent's thought process (this replaces thought_log)"""
        result = ""
        for action, observation in intermediate_steps:
            result += action.log 
            result += f"\nObservation: {observation}\n"
        return result

    def format(self, **kwargs) -> str:
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
                    "thought_process": self.format_scratchpad(
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

            kwargs["agent_scratchpad"] += "Here go your thoughts and actions:\n"

            current_thought_steps = intermediate_steps[-self.current_context_length:]
            kwargs["agent_scratchpad"] += self.format_scratchpad(current_thought_steps)
            
        if 'input' not in kwargs:
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
        result = remove_surrogates(
            remove_project_summaries(self.template.format(**kwargs)))
        result = trim_extra(result, 25000)
        if self.hook:
            self.hook(self)
        if self.project and os.path.exists(self.project.path):
            with open(os.path.join(self.project.path, ".prompts.log"), "a") as f:
                f.write(result + "\n\n============================\n\n\n")
        return result


@dataclass
class BaseMinion:
    def __init__(
            self,
            base_prompt,
            available_tools,
            max_iterations: int = 50,
            allow_feedback: bool = False,
            max_context_length: int = 10,
            keep_n_last_thoughts: int = 2
    ) -> None:
        try:
            llm = CustomLlamaCliLLM()
        except ValueError as e:
            logger.error(f"Failed to initialize CustomLlamaCliLLM in BaseMinion: {e}")
            raise e
        agent_toolnames = [tool.name for tool in available_tools]
        extended_tools = list(available_tools)
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

        agent = create_react_agent(
            llm=llm,
            tools=available_tools,
            prompt=self.prompt
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=available_tools,
            verbose=True,
            handle_parsing_errors=custom_handle_parsing_errors, 
            max_iterations=10,
            early_stopping_method="generate", 
            return_intermediate_steps=True    
        )
        self.allow_feedback = allow_feedback

    def invoke(self, inputs: dict) -> dict:
        inputs["feedback"] = inputs.get("feedback", "")
        inputs["format_description"] = format_description
        
        return run_with_retries(self.agent_executor, inputs)


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
    ) -> None:
        try:
            eval_llm_instance = CustomLlamaCliLLM()
        except ValueError as e:
            logger.error(f"Failed to initialize CustomLlamaCliLLM in FeedbackMinion for eval_llm: {e}")
            raise e
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
            try:
                current_feedback_text = original_inputs.get("feedback", "")
                previous_result_text = original_inputs.get("previous_result", "")
                feedback_format_args = {
                    "feedback": current_feedback_text,
                    "previous_result": previous_result_text
                }
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
