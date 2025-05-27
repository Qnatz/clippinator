import os
from crewai_tools import tool
from datetime import datetime

@tool("Get Project Summary Tool")
def get_project_summary_tool(path: str = ".") -> str:
    """
    Provides a summary of the project directory structure and files.
    Lists files and subdirectories. For text files, it may show a few initial lines.
    Input: Optional path string, defaults to current directory.
    """
    summary_lines = []
    try:
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            return f"Error: Path '{abs_path}' is not a valid directory."

        summary_lines.append(f"Project Summary for directory: {abs_path}\n")
        
        for root, dirs, files in os.walk(abs_path):
            # Skip .git, __pycache__, and other common non-project dirs
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
            
            level = root.replace(abs_path, '').count(os.sep)
            indent = "    " * level
            # Ensure that the root itself is not added if it's the starting path, to avoid double listing
            if root != abs_path:
                 summary_lines.append(f"{indent}{os.path.basename(root)}/")
            elif not summary_lines[-1].strip().endswith("/"): # If it's the first line, ensure it ends with /
                 summary_lines[-1] = summary_lines[-1].strip() + "/"


            sub_indent = "    " * (level + 1)
            for f_name in files:
                # Skip common non-project files
                if f_name in ['.DS_Store', '.gitignore', '.env', 'PROJECT_MEMORIES.log']: # Added PROJECT_MEMORIES.log
                    continue
                summary_lines.append(f"{sub_indent}{f_name}")
                # Optionally, add logic here to read first few lines of text files
                # For simplicity, this version just lists names.

        output = "\n".join(summary_lines)
        if len(output) > 7000: # Max output length
            output = output[:7000] + "\n... (summary truncated)"
        return output
    except Exception as e:
        return f"Error getting project summary for path '{path}': {str(e)}"

MEMORY_FILE = "PROJECT_MEMORIES.log"

@tool("Remember Fact Tool")
def remember_fact_tool(fact: str) -> str:
    """
    Records a fact or piece of information into a shared project memory log (PROJECT_MEMORIES.log).
    Input: A string representing the fact to remember.
    """
    if not isinstance(fact, str):
        return "Error: The fact to remember must be a string."
    if not fact.strip():
        return "Error: Cannot remember an empty fact."
        
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp}: {fact.strip()}\n"
        
        # Log will be created in the current working directory of the script/agent
        with open(MEMORY_FILE, "a") as f:
            f.write(log_entry)
        return f"Fact remembered and logged to {MEMORY_FILE}."
    except Exception as e:
        return f"Error remembering fact: {str(e)}"
