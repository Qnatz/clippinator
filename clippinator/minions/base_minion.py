from __future__ import annotations

import os
import re
import logging # Added import
from dataclasses import dataclass
from typing import List, Union, Callable, Any

import langchain.schema
from langchain.chains import LLMChain # Updated import
from langchain.prompts import PromptTemplate # Updated import
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
    # max_context_length: int = 5 # Now an instance variable
    # keep_n_last_thoughts: int = 2 # Now an instance variable
    current_context_length: int = 0
    model_steps_processed: int = 0
    all_steps_processed: int = 0
    my_summarize_agent: Any = None
    last_summary: str = ""
    project: Any | None = None
    intermediate_steps: list = [] # Field for intermediate steps
    hook: Optional[Callable[[CustomPromptTemplate], None]] = None

    def __init__(self, **kwargs: Any):
        # Pydantic v1: Pass all relevant kwargs to super().__init__().
        # Fields defined on CustomPromptTemplate (template, tools, agent_toolnames)
        # and fields expected by StringPromptTemplate (input_variables)
        # will be picked up by Pydantic's initialization mechanism.
        super().__init__(**kwargs)

        # Initialize attributes that are not direct Pydantic fields passed to super(),
        # or that need specific default logic not handled by class var defaults + Pydantic.
        # For attributes that *are* Pydantic fields (like template, tools, agent_toolnames, input_variables),
        # they are already set by super().__init__(**kwargs) if they were in kwargs.

        # These seem like configurable parameters rather than core Pydantic fields of the prompt template itself.
        # If they were meant to be fields, they should be passed to super() too.
        # For now, setting them as instance attributes from kwargs or defaults.
        self.max_context_length = kwargs.get("max_context_length", 5)
        self.keep_n_last_thoughts = kwargs.get("keep_n_last_thoughts", 2)
        
        # project, my_summarize_agent, hook are passed during instantiation and should be set on self.
        # If they are also meant to be Pydantic fields, they should be part of kwargs to super().
        # Assuming they are just instance attributes for now:
        self.project = kwargs.get("project")
        self.my_summarize_agent = kwargs.get("my_summarize_agent")
        self.hook = kwargs.get("hook")

        # Initialize mutable state attributes here if not handled by Pydantic Field(default_factory=list)
        # For Pydantic fields, super().__init__(**kwargs) would have initialized them if they were in kwargs,
        # or used their class-level defaults (including default_factory).
        # If 'intermediate_steps' is a class-annotated field (e.g., `intermediate_steps: list = Field(default_factory=list)`),
        # Pydantic handles its default initialization.
        # If it's just a regular instance attribute not declared as a Pydantic field, initialize it here.
        # Based on its usage, it's state, so ensure it's initialized.
        if not hasattr(self, 'intermediate_steps') or self.intermediate_steps is None: # Check if Pydantic set it
            self.intermediate_steps = []
        
        # These are likely already initialized by their class variable defaults if not in kwargs.
        # No need to re-initialize unless specific logic is needed.
        # self.current_context_length = 0 # Already a class var default
        # self.model_steps_processed = 0 # Already a class var default
        # self.all_steps_processed = 0 # Already a class var default
        # self.last_summary = "" # Already a class var default

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
