from crewai import Agent
from ..tools.file_io_tools import read_file_tool, write_file_tool
from ..tools.command_tools import shell_command_tool # For running test scripts
# from ..tools.python_tools import python_script_tool # If we create a dedicated python script tool

def create_qa_agent(llm_instance=None):
    return Agent(
        role="Diligent QA Engineer and Software Tester",
        goal="To ensure software quality by meticulously reviewing code, writing and executing tests, "
             "and identifying bugs. Verify that implemented features meet requirements and the code is robust, "
             "paying close attention to edge cases and potential failure points.",
        backstory=(
            "You are a highly skilled QA engineer with a strong background in software testing and quality assurance. "
            "You have a keen eye for detail and a methodical approach to testing. "
            "You are proficient in creating test plans, writing test cases (including unit, integration, and functional tests), "
            "and clearly documenting any bugs or issues found. Your goal is to ensure the software is as bug-free "
            "and reliable as possible before it's considered complete."
        ),
        tools=[read_file_tool, write_file_tool, shell_command_tool], # Add python_script_tool if available
        llm=llm_instance,
        allow_delegation=False, # QA agent focuses on testing, not delegating tests
        verbose=True
    )
