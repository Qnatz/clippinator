from __future__ import annotations

import os
import re
import logging # Added import
from dataclasses import dataclass
from typing import List, Union, Callable, Any, Optional # Added Optional

import langchain.schema
from langchain.chains.llm import LLMChain # Updated import
from langchain_core.prompts import PromptTemplate # Updated import
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
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


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        actions = [
            line.split(":", 1)[1].strip()
            for line in llm_output.splitlines()
            if line.startswith("Action:")
        ]
        # Check if agent should finish"
        if "Final Result:" in llm_output:
            if "Action" in llm_output:
                return AgentAction(
                    tool="WarnAgent",
                    tool_input=f"ERROR: Don't write 'Action' together with the Final Result. "
                               f"You need to REDO your action(s) ({', '.join(actions)}), "
                               f"receive the 'AResult' and only then write your 'Final Result'",
                    log=llm_output,
                )
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Result:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match and llm_output.strip().split("\n")[-1].strip().startswith(
                "Thought:"
        ):
            return AgentAction(
                tool="WarnAgent",
                tool_input="don't stop after 'Thought:', continue with the next thought or action",
                log=llm_output,
            )

        if not match:
            if "Action:" in llm_output and "\nAction Input:" not in llm_output:
                return AgentAction(
                    tool="WarnAgent",
                    tool_input="No Action Input specified.",
                    log=llm_output,
                )
            else:
                return AgentAction(
                    tool="WarnAgent",
                    tool_input="Continue with your next thought or action. Do not repeat yourself. "
                               "When you're done, write 'Final Result:'. \n",
                    log=llm_output,
                )

        if llm_output.count("\nAction Input:") > 1:
            return AgentAction(
                tool="WarnAgent",
                tool_input="ERROR: Write 'AResult: ' after each action. Execute ALL the past actions "
                           f"without AResult again ({', '.join(actions)}), one-by-one. They weren't completed.",
                log=llm_output,
            )

        action = match.group(1).strip().strip("`").strip('"').strip("'").strip()
        action_input = match.group(2)
        if "\nThought: " in action_input or "\nAction: " in action_input:
            return AgentAction(
                tool="WarnAgent",
                tool_input="Error: Write 'AResult: ' after each action. "
                           f"Execute all the actions without AResult again ({', '.join(actions)}).",
                log=llm_output,
            )
        if "Subagent" in action:
            action_input += " " + action.split("Subagent")[1].strip()
            action = "Subagent"

        # Return the action and action inputx
        return AgentAction(
            tool=action,
            tool_input=action_input.strip(" ").split("\nThought: ")[0],
            log=llm_output,
        )


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
    prompt: PromptTemplate
    llm: LLMChain

    def __init__(self, base_prompt: str) -> None: # MODIFIED SIGNATURE
        try:
            llm = CustomLlamaCliLLM() # NEW INSTANTIATION
        except ValueError as e:
            logger.error(f"Failed to initialize CustomLlamaCliLLM in BasicLLM: {e}")
            raise e # Re-raise the exception after logging
        self.llm = LLMChain( # REMAINS THE SAME
            llm=llm,
            prompt=PromptTemplate(
                template=base_prompt,
                input_variables=extract_variable_names(base_prompt),
            ),
        )

    def run(self, **kwargs):
        kwargs["feedback"] = kwargs.get("feedback", "")
        return self.llm.predict(**kwargs)


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
    intermediate_steps: list = [] # Will be initialized in __init__
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
        
        self.intermediate_steps: list = []

    @property
    def _prompt_type(self) -> str:
        return "taskmaster"

    def thought_log(self, thoughts: list[(AgentAction, str)]) -> str:
        result = ""
        for i, (action, aresult) in enumerate(thoughts):
            if self.my_summarize_agent:
                aresult = trim_extra(aresult, 1300 if i != len(thoughts) - 1 else 1750)
            if action.tool == "WarnAgent":
                result += action.log + f"\nSystem note: {aresult}\n"
            elif action.tool == "AgentFeedback":
                result += action.log + aresult + "\n"
            else:
                result += action.log + f"\nAResult: {aresult}\n"
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
                self.last_summary = self.my_summarize_agent.run(
                    summary=self.last_summary,
                    thought_process=self.thought_log(
                        intermediate_steps[
                        -self.current_context_length: -self.keep_n_last_thoughts
                        ]
                    ),
                )
                self.current_context_length = self.keep_n_last_thoughts

            if self.my_summarize_agent:
                kwargs["agent_scratchpad"] = (
                        "Here is a summary of what has happened:\n" + trim_extra(self.last_summary, 2700, 1900)
                )
                kwargs["agent_scratchpad"] += "\nEND OF SUMMARY\n"
            else:
                kwargs["agent_scratchpad"] = ""

            kwargs["agent_scratchpad"] += "Here go your thoughts and actions:\n"

            kwargs["agent_scratchpad"] += self.thought_log(
                intermediate_steps[-self.current_context_length:]
            )

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
            remove_project_summaries(self.template.format(**kwargs).replace('{tools}', kwargs['tools'])))
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
            max_iterations: int = 50,
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

        llm_chain = LLMChain(llm=llm, prompt=self.prompt)
        output_parser = CustomOutputParser()
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["AResult:"],
            allowed_tools=[tool.name for tool in available_tools], # Original list, not extended
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=available_tools, # Original list
            verbose=True,
            max_iterations=max_iterations,
        )
        self.allow_feedback = allow_feedback

    def run(self, **kwargs):
        kwargs["feedback"] = kwargs.get("feedback", "")
        kwargs["format_description"] = format_description
        if not self.allow_feedback:
            return (
                    self.agent_executor.run(**kwargs)
                    or "No result. The execution was probably unsuccessful."
            )
        try:
            return (
                    self.agent_executor.run(**kwargs)
                    or "No result. The execution was probably unsuccessful."
            )
        except KeyboardInterrupt:
            feedback = ask_for_feedback()
            if feedback:
                self.prompt.intermediate_steps += [feedback]
            return self.run(**kwargs)

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

    def run(self, **kwargs):
        if "feedback" in kwargs:
            print("Rerunning a prompt with feedback:", kwargs["feedback"])
            if len(kwargs["previous_result"]) > 500:
                kwargs["previous_result"] = (
                        kwargs["previous_result"][:500] + "\n...(truncated)\n"
                )
            kwargs["feedback"] = self.feedback_prompt.format(**kwargs)
        res = self.underlying_minion.run(**kwargs)
        try:
            check_result = None # Initialize check_result to None
            self.check_function(res)
        except ValueError as e:
            check_result = " ".join(e.args)
        if check_result: # This condition was using an undefined variable `critique`
            kwargs["feedback"] = check_result
            kwargs["previous_result"] = res
            return self.run(**kwargs)
        evaluation = self.eval_llm.predict(result=res, **kwargs)
        if "ACCEPT" in evaluation:
            return res
        kwargs["feedback"] = evaluation.split("Feedback: ", 1)[-1].strip()
        kwargs["previous_result"] = res
        return self.run(**kwargs)
