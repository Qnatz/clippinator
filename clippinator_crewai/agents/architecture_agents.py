from crewai import Agent
from ..tools.file_io_tools import read_file_tool, write_file_tool
# from ..tools.web_tools import web_browser_tool # Assuming web_browser_tool is or will be defined elsewhere
# For now, let's assume a generic web search tool might be added later or is available.
# If SimpleWebBrowserTool was adapted, it would be imported.
# For this step, basic file tools are primary.

def create_software_architect_agent(llm_instance=None):
    return Agent(
        role="Lead Software Architect",
        goal="To design robust and scalable software architecture based on the project plan and objectives. "
             "Define file structures, key components, classes, functions, and their interactions. "
             "Ensure the architecture aligns with best practices and project requirements. "
             "The final architecture should be saved to a file named PROJECT_ARCHITECTURE.md.",
        backstory=(
            "You are a visionary software architect with a deep understanding of various design patterns, "
            "technologies, and system integration. You can create clear and detailed architectural blueprints "
            "that guide the development team effectively. You will receive a project plan "
            "and must produce a markdown file named PROJECT_ARCHITECTURE.md detailing the software architecture."
        ),
        tools=[read_file_tool, write_file_tool], # Add web_browser_tool or research tools later if needed
        llm=llm_instance,
        allow_delegation=False,
        verbose=True
    )
