import os
import sys
import yaml
import importlib
from pathlib import Path
from crewai import Crew, Agent, Task, Process
from langchain.tools import Tool # For wrapping functions if needed for descriptions

# --- Path Setup ---
# Ensure 'src' is in the Python path to allow imports like 'from shared_tools...'
# This assumes the script is run from the project root where 'src' is a subdirectory.
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASE_DIR) not in sys.path: # Also add base if flow.yaml is at root
    sys.path.insert(0, str(BASE_DIR))

# --- Helper function to dynamically import tool functions ---
def import_tool_function(tool_path_string: str):
    module_path, func_name = tool_path_string.rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    except (ImportError, AttributeError) as e:
        print(f"Error importing tool '{tool_path_string}': {e}")
        return None

def run_yaml_driven_crew():
    print("Initializing YAML-driven development crew...")

    # --- Load Main Flow Configuration ---
    flow_file_path = BASE_DIR / "coding_agents_flow.yaml"
    try:
        with open(flow_file_path, "r") as f:
            flow_config = yaml.safe_load(f)
        print(f"Loaded flow configuration from: {flow_file_path}")
    except Exception as e:
        print(f"Error loading coding_agents_flow.yaml: {e}")
        return

    # --- LLM Configuration (from previous setup) ---
    llm_choice = os.getenv("LLM_CHOICE", "openai").lower()
    global_llm_config_str = None # This will be the model string for agents

    print(f"LLM choice: '{llm_choice}' (set via LLM_CHOICE environment variable or defaulted).")
    if llm_choice == "ollama":
        ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-coder")
        global_llm_config_str = f"ollama/{ollama_model}"
        print(f"Configured to use Ollama model: '{global_llm_config_str}'. Ensure OLLAMA_MODEL and OLLAMA_BASE_URL (if not default) are set.")
    elif llm_choice == "gemini":
        gemini_model = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
        global_llm_config_str = f"gemini/{gemini_model}"
        print(f"Configured to use Gemini model: '{global_llm_config_str}'. Ensure GEMINI_MODEL_NAME and GOOGLE_API_KEY are set.")
    elif llm_choice == "openai":
        openai_model = os.getenv("OPENAI_MODEL_NAME")
        if openai_model:
            global_llm_config_str = openai_model
            print(f"Configured to use OpenAI model: '{global_llm_config_str}'.")
        else:
            print("Configured for default OpenAI handling by CrewAI/LiteLLM (llm=None for agent).")
        print("Ensure OPENAI_API_KEY is set. Optionally, set OPENAI_MODEL_NAME.")
    else:
        print(f"Warning: LLM_CHOICE '{llm_choice}' not recognized. CrewAI will attempt its default LLM resolution.")
    # --- End LLM Configuration ---

    # --- Instantiate Agents ---
    instantiated_agents = {}
    print("Instantiating agents...")
    for agent_name_in_flow in flow_config.get("agents", []):
        agent_config_path = SRC_DIR / agent_name_in_flow / "config" / "agents.yaml"
        try:
            with open(agent_config_path, "r") as f:
                agent_yaml_data = yaml.safe_load(f)
            
            agent_info = agent_yaml_data["agents"][0] 
            
            agent_tools = []
            for tool_path_str in agent_info.get("tools", []):
                tool_func = import_tool_function(tool_path_str)
                if tool_func:
                    agent_tools.append(tool_func) 
            
            agent_llm_str = agent_info.get("llm") or global_llm_config_str

            agent = Agent(
                role=agent_info["role"],
                goal=agent_info["goal"],
                backstory=agent_info.get("backstory", ""),
                llm=agent_llm_str, 
                tools=agent_tools,
                verbose=agent_info.get("verbose", True),
                allow_delegation=agent_info.get("allow_delegation", False)
            )
            instantiated_agents[agent_name_in_flow] = agent
            print(f"  Agent '{agent_name_in_flow}' instantiated with role: {agent.role}")
        except FileNotFoundError:
            print(f"Warning: Agent config file not found for '{agent_name_in_flow}' at {agent_config_path}. Skipping this agent.")
        except Exception as e:
            print(f"Error instantiating agent '{agent_name_in_flow}' from {agent_config_path}: {e}")
            # Decide if to stop or continue without this agent
            # For now, we'll print a warning and it will be missing from instantiated_agents

    # --- Instantiate Tasks (Simplified Sequential Order for now) ---
    instantiated_tasks = []
    print("Instantiating tasks...")
    task_objects = {} # Using task_objects as per the provided snippet

    for task_def in flow_config.get("tasks", []):
        task_id = task_def["id"]
        agent_name = task_def["agent"]
        assigned_agent = instantiated_agents.get(agent_name)

        if not assigned_agent:
            print(f"Error: Agent '{agent_name}' for task '{task_id}' not found or failed to instantiate. Skipping task.")
            continue
        
        task = Task(
            description=task_def["description"],
            expected_output=task_def.get("expected_output", "No specific expected output defined."),
            agent=assigned_agent,
        )
        task_objects[task_id] = task # Store task object by its ID
    
    for task_def in flow_config.get("tasks", []):
        task_id = task_def["id"]
        if task_id in task_objects:
             instantiated_tasks.append(task_objects[task_id])
             print(f"  Task '{task_id}' ({task_objects[task_id].description[:30]}...) added to sequence.")

    # --- Assemble and Run Crew ---
    if not instantiated_agents or not instantiated_tasks:
        print("No agents or tasks were successfully instantiated. Cannot run crew.")
        return

    print("Assembling crew...")
    # Filter out agents that were not successfully instantiated if some failed but we decided to continue
    active_agents_in_crew = [agent for agent_name, agent in instantiated_agents.items() if any(task.agent == agent for task in instantiated_tasks)]


    crew = Crew(
        agents=active_agents_in_crew, # Use only agents that have tasks
        tasks=instantiated_tasks, 
        process=Process.sequential, 
        verbose=2
    )

    print("Crew assembled. Kicking off execution...")
    result = crew.kickoff()

    print("\n\n--- YAML-Driven Crew Execution Result ---")
    print(result)
    print("--- End of Crew Execution ---")

    print("\n--- File Verification (example) ---")
    files_to_check_for_existence = ["PROJECT_PLAN.md", "PROJECT_ARCHITECTURE.md"] 
    for f_name in files_to_check_for_existence:
        file_path = BASE_DIR / f_name 
        if file_path.exists():
            print(f"File '{f_name}' was created successfully at: {file_path}")
        else:
            print(f"File '{f_name}' was NOT created at: {file_path}")


if __name__ == "__main__":
    print(f"Executing YAML-driven crew from CWD: {os.getcwd()}")
    print("Ensure CrewAI is installed (`pip install crewai crewai-tools pyyaml langchain langchain-community langchain-google-genai langchain-openai`)")
    print("Configure LLM via environment variables (see script comments).")
    run_yaml_driven_crew()
