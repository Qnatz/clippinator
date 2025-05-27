from crewai import Agent
# crewai_tools.BaseTool is not explicitly needed if we are just using tools
# imported from other modules that are already decorated with @tool.

# Relative imports as specified, assuming the execution context allows this structure.
# (e.g., main_crew_runner.py is in clippinator_crewai/ or the project is installed in a way
# that makes clippinator_crewai a package).
from ..tools.file_io_tools import read_file_tool, write_file_tool
from ..tools.command_tools import shell_command_tool

# Optional: Define a specific LLM for testing or default use.
# For this task, we'll allow it to be passed in, defaulting to None.
# from langchain_openai import ChatOpenAI # Example if you wanted to define one here
# default_llm = ChatOpenAI(model="gpt-3.5-turbo") # Example

def create_code_writer_agent(llm_instance=None):
    """
    Creates the CodeWriterAgent.
    
    Args:
        llm_instance: An optional pre-configured LLM instance from a library like
                      langchain_openai, langchain_anthropic, etc.
                      If None, CrewAI will use its default LLM or one configured 
                      globally for the Crew.
    
    Returns:
        A CrewAI Agent instance configured as the CodeWriterAgent.
    """
    return Agent(
        role="Expert Software Developer",
        goal="Write clean, efficient, and well-documented code based on specifications and assigned tasks. "
             "Implement features, fix bugs, and ensure code quality. "
             "You must create the specified files with the correct content.",
        backstory=(
            "You are a proficient software engineer with expertise in multiple programming languages and frameworks. "
            "You follow coding standards rigorously and are adept at turning specifications into functional code. "
            "You are detail-oriented and ensure that your code is not only functional but also readable and maintainable. "
            "When asked to create a file, you focus solely on creating that file with the exact content requested."
        ),
        tools=[read_file_tool, write_file_tool, shell_command_tool],
        llm=llm_instance,  # Assign the LLM instance here
        allow_delegation=False, # This agent focuses on its writing task and does not delegate.
        verbose=True # Enable verbose output for easier debugging and observation
    )

# Example of how this agent might be instantiated in a crew setup script:
#
# from langchain_openai import ChatOpenAI # Or any other LLM provider
#
# # Initialize your desired LLM
# my_llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7)
#
# # Create the code writer agent using the factory function
# code_writer_agent = create_code_writer_agent(llm_instance=my_llm)
#
# # This agent would then be added to a Crew.
# # from crewai import Crew
# # my_crew = Crew(agents=[code_writer_agent], tasks=[...], ...)
# # result = my_crew.kickoff()
#
# # If no LLM is passed, CrewAI's default will be used:
# # default_code_writer_agent = create_code_writer_agent()
