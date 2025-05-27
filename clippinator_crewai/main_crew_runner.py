import os
import sys
from crewai import Crew, Process
import re # For creating the expected report filename for verification

# LLM Client Imports
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI # Explicit import for clarity, even if CrewAI handles default

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

    # --- LLM Configuration ---
    # Choose your LLM provider here by uncommenting one section and commenting others.
    # Ensure you have the necessary environment variables set for your chosen LLM.

    llm_choice = "openai" # Options: "openai", "ollama", "gemini" 
    # To use a specific LLM, change "openai" to "ollama" or "gemini".
    # Ensure the chosen LLM is configured (e.g., Ollama server running, API keys set).
    llm_to_use = None

    print(f"Attempting to configure LLM: {llm_choice}")

    if llm_choice == "ollama":
        # Ensure Ollama server is running.
        # Replace "deepseek-coder" with the actual model name if different (e.g., "deepseek-coder:6.7b")
        # You might need to set OLLAMA_BASE_URL environment variable if not default.
        try:
            # Check for OLLAMA_MODEL environment variable, otherwise default to deepseek-coder
            ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-coder")
            llm_to_use = ChatOllama(
                model=ollama_model, 
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            print(f"Using Ollama with model '{ollama_model}'. Ensure Ollama is running and the model is pulled/served.")
        except Exception as e:
            print(f"Error initializing Ollama: {e}. Proceeding without specific LLM (will use CrewAI default or fail).")
            llm_to_use = None

    elif llm_choice == "gemini":
        # Ensure GOOGLE_API_KEY environment variable is set.
        try:
            gemini_model = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
            llm_to_use = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.7)
            print(f"Using Google Gemini with model '{gemini_model}'.")
        except Exception as e:
            print(f"Error initializing Gemini: {e}. Proceeding without specific LLM.")
            llm_to_use = None
            
    elif llm_choice == "openai":
        # Ensure OPENAI_API_KEY is set. Optionally OPENAI_MODEL_NAME.
        try:
            # If OPENAI_API_KEY is set, CrewAI agents will use OpenAI by default if llm_instance is None.
            # To explicitly specify a model or other parameters for OpenAI:
            # llm_to_use = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"), temperature=0.7)
            # For this example, we'll let CrewAI handle OpenAI defaults if llm_to_use is None.
            print("Using OpenAI. CrewAI will use its default OpenAI handling (ensure OPENAI_API_KEY is set).")
            # If you want to explicitly set it (e.g. to a specific model):
            # openai_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")
            # llm_to_use = ChatOpenAI(model=openai_model) 
            # print(f"Explicitly set OpenAI model to: {openai_model}")
            pass # llm_to_use remains None, CrewAI handles OpenAI if API key is present and no other LLM is specified.
        except Exception as e:
            print(f"Error initializing OpenAI (explicitly): {e}. Proceeding with CrewAI default LLM handling.")
            llm_to_use = None # Fallback to CrewAI default
    else:
        print(f"LLM choice '{llm_choice}' is invalid or not explicitly handled. CrewAI will use its default LLM.")
        llm_to_use = None
    # --- End LLM Configuration ---

    # 1. Create Agents (pass the chosen llm_to_use)
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
    print("Ensure CrewAI is installed (`pip install crewai crewai-tools langchain-community langchain-google-genai langchain-openai`)")
    print("LLM Choice: Set the 'llm_choice' variable in the script to 'ollama', 'gemini', or 'openai'.")
    print("Ensure the chosen LLM is configured (e.g., Ollama server running, API keys like GOOGLE_API_KEY or OPENAI_API_KEY set).")
    
    run_development_crew()
