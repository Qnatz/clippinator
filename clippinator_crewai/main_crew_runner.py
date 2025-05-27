import os
import sys
from crewai import Crew, Process
import re # For creating the expected report filename for verification

# LLM Client Imports are no longer needed here for explicit instantiation.
# CrewAI will handle LLM resolution via LiteLLM.

# Adjust sys.path for imports
current_script_path = os.path.abspath(__file__)
clippinator_crewai_dir = os.path.dirname(current_script_path)
project_root_dir = os.path.dirname(clippinator_crewai_dir)

if clippinator_crewai_dir not in sys.path:
    sys.path.append(clippinator_crewai_dir)
if project_root_dir not in sys.path:
     sys.path.append(project_root_dir)


try:
    from agents.coding_agents import create_code_writer_agent
    from agents.planning_agents import create_project_planner_agent
    from agents.architecture_agents import create_software_architect_agent
    from agents.qa_agents import create_qa_agent
    from tasks.planning_tasks import create_initial_planning_task
    from tasks.architecture_tasks import create_architecture_design_task
    from tasks.coding_tasks import create_coding_task
    from tasks.qa_tasks import create_testing_task
except ImportError as e:
    print(f"Import error: {e}. CWD: {os.getcwd()}")
    print(f"sys.path: {sys.path}")
    print("Please ensure PYTHONPATH is set correctly or run from a directory that allows these imports.")
    sys.exit(1)

def run_development_crew():
    print("Initializing development crew...")

    # --- Simplified LLM Configuration using CrewAI's LiteLLM integration ---
    # Set the LLM_CHOICE environment variable to "ollama", "gemini", or "openai".
    # Default is "openai" if LLM_CHOICE is not set.
    llm_choice = os.getenv("LLM_CHOICE", "openai").lower()
    llm_to_use = None # This will be the model string or None

    print(f"LLM choice: '{llm_choice}' (set via LLM_CHOICE environment variable or defaulted).")

    if llm_choice == "ollama":
        # User needs to ensure OLLAMA_MODEL is set to the model they serve (e.g., "deepseek-coder:6.7b")
        # and OLLAMA_BASE_URL if not http://localhost:11434
        ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-coder") 
        llm_to_use = f"ollama/{ollama_model}" # CrewAI/LiteLLM format
        print(f"Configured to use Ollama model: '{llm_to_use}'.")
        print("Ensure OLLAMA_MODEL environment variable is set to your specific model (e.g., 'deepseek-coder:6.7b').")
        print("Ensure OLLAMA_BASE_URL environment variable is set if your Ollama instance is not at http://localhost:11434.")
    elif llm_choice == "gemini":
        # User needs to ensure GEMINI_MODEL_NAME is set (e.g., "gemini-1.5-flash-latest")
        # and GOOGLE_API_KEY is set.
        gemini_model = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
        llm_to_use = f"gemini/{gemini_model}" # CrewAI/LiteLLM format
        print(f"Configured to use Gemini model: '{llm_to_use}'.")
        print("Ensure GEMINI_MODEL_NAME environment variable is set to your specific model (e.g., 'gemini-1.5-flash-latest').")
        print("Ensure GOOGLE_API_KEY environment variable is set.")
    elif llm_choice == "openai":
        # User needs to ensure OPENAI_API_KEY is set.
        # Optionally, set OPENAI_MODEL_NAME to specify a model (e.g., "gpt-4o-mini").
        # If OPENAI_MODEL_NAME is not set, CrewAI/LiteLLM uses a default (like gpt-4o-mini).
        openai_model = os.getenv("OPENAI_MODEL_NAME")
        if openai_model:
            llm_to_use = openai_model # Directly use the model name string
            print(f"Configured to use OpenAI model: '{llm_to_use}'.")
        else:
            # llm_to_use remains None. CrewAI/LiteLLM will use its default OpenAI model
            # if OPENAI_API_KEY is available.
            print("Configured for default OpenAI handling by CrewAI/LiteLLM.")
        print("Ensure OPENAI_API_KEY environment variable is set.")
        print("Optionally, set OPENAI_MODEL_NAME to specify a model (e.g., 'gpt-4o-mini', 'gpt-4-turbo').")
    else:
        print(f"Warning: LLM_CHOICE '{llm_choice}' not recognized. CrewAI will attempt its default LLM resolution (likely OpenAI if configured).")
        # llm_to_use remains None, relying on CrewAI's global default or erroring if none configured.
    # --- End LLM Configuration ---

    # 1. Create Agents (pass the chosen llm_to_use, which is now a string or None)
    # The agent creation functions should pass this directly to the Agent's 'llm' parameter.
    planner_agent = create_project_planner_agent(llm_instance=llm_to_use)
    architect_agent = create_software_architect_agent(llm_instance=llm_to_use)
    coder_agent = create_code_writer_agent(llm_instance=llm_to_use)
    qa_agent = create_qa_agent(llm_instance=llm_to_use)
    print(f"Agents created. Planner: {planner_agent.role}, Architect: {architect_agent.role}, Coder: {coder_agent.role}, QA: {qa_agent.role}")

    # 2. Define Tasks
    project_objective = "Create a simple Python CLI application that takes a user's name as an argument and prints a personalized greeting. The application should have a main function and be runnable from the command line."

    planning_task = create_initial_planning_task(
        agent_to_assign=planner_agent,
        objective=project_objective
    )
    print(f"Planning task created: {planning_task.description[:50]}...")

    architecture_task = create_architecture_design_task(
        agent_to_assign=architect_agent
    )
    architecture_task.context = [planning_task] 
    print(f"Architecture task created: {architecture_task.description[:50]}...")

    coding_task_description = (
        "Implement the main CLI application file as specified in 'PROJECT_ARCHITECTURE.md'. "
        "This file should be named appropriately (e.g., 'main_cli.py' or as per architecture output). "
        "It should include argument parsing for a 'name' and a function that prints the greeting. "
        "Ensure the file is created in the current working directory."
    )
    coding_task_filename = "main_cli_app.py" 
    
    coding_task = create_coding_task(
        filename=coding_task_filename,
        description=coding_task_description,
        agent_to_assign=coder_agent
    )
    coding_task.context = [architecture_task] 
    print(f"Coding task created: {coding_task.description[:50]}...")

    safe_report_base = re.sub(r'[^a-zA-Z0-9_.-]', '_', coding_task_filename)
    expected_qa_report_filename = f"TEST_REPORT_{safe_report_base.replace('/', '_').replace('.', '_')}.md"

    testing_task_requirements = (
        "Verify that the CLI application correctly parses the 'name' argument "
        "and prints a personalized greeting. Check for basic error handling if no name is provided. "
        "Ensure the application is runnable and produces the expected output."
    )
    testing_task = create_testing_task(
        agent_to_assign=qa_agent,
        file_to_test=coding_task_filename, 
        test_requirements=testing_task_requirements
    )
    testing_task.context = [coding_task] 
    print(f"Testing task created: {testing_task.description[:50]}...")

    # 3. Create the Crew
    development_crew = Crew(
        agents=[planner_agent, architect_agent, coder_agent, qa_agent],
        tasks=[planning_task, architecture_task, coding_task, testing_task],
        process=Process.sequential, 
        verbose=2 
    )
    print("Crew configured. Kicking off execution...")

    # 4. Run the Crew
    result = development_crew.kickoff()

    print("\n\n--- Crew Execution Result ---")
    print(result) 
    print("--- End of Crew Execution ---")

    # Verify file creation
    print("\n--- File Verification ---")
    current_working_dir = os.getcwd()
    files_to_check = [
        "PROJECT_PLAN.md", 
        "PROJECT_ARCHITECTURE.md", 
        coding_task_filename, 
        expected_qa_report_filename
    ]
    
    for f_name in files_to_check:
        file_path = os.path.join(current_working_dir, f_name)
        if os.path.exists(file_path):
            print(f"File '{f_name}' was created successfully at: {file_path}")
            try:
                with open(file_path, "r") as f_content:
                    content_preview = f_content.read(300) 
                    print(f"Content preview of '{f_name}':\n{content_preview}...\n")
            except Exception as e:
                print(f"Could not read content of '{f_name}': {e}\n")
        else:
            print(f"File '{f_name}' was NOT created at: {file_path}")
    print(f"Note: Files are checked in CWD: {current_working_dir}")

if __name__ == "__main__":
    print(f"Executing main_crew_runner.py from CWD: {os.getcwd()}")
    print("Ensure CrewAI is installed (`pip install crewai crewai-tools`)")
    print("For LLM configuration, set the following environment variables as needed:")
    print("  - LLM_CHOICE: 'ollama', 'gemini', or 'openai' (defaults to 'openai').")
    print("  - For Ollama: OLLAMA_MODEL (e.g., 'deepseek-coder:6.7b'), OLLAMA_BASE_URL (if not default).")
    print("  - For Gemini: GEMINI_MODEL_NAME (e.g., 'gemini-1.5-flash-latest'), GOOGLE_API_KEY.")
    print("  - For OpenAI: OPENAI_API_KEY, optionally OPENAI_MODEL_NAME (e.g., 'gpt-4o-mini').")
    
    run_development_crew()
