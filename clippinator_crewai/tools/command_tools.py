from crewai_tools import tool
import subprocess
import os

@tool("Executes a shell command and returns its output. Input must be the command string.")
def shell_command_tool(command: str) -> str:
    """
    Executes a shell command and returns its output.
    Input:
        command (str): The shell command to execute.
    """
    if not isinstance(command, str):
        return "Error: Command must be a string."
    try:
        # For security and simplicity, run in the current working directory of the script.
        # Timeout is important for agents.
        completed_process = subprocess.run(
            command,
            shell=True, # Be cautious with shell=True if command comes from LLM without sanitization
            capture_output=True,
            text=True,
            timeout=60, # 60-second timeout
            cwd=os.getcwd() # Explicitly set CWD, though it's often the default
        )
        output = completed_process.stdout if completed_process.stdout else ""
        error_output = completed_process.stderr if completed_process.stderr else ""

        if error_output:
            if output: # if there's also stdout, append stderr
                 output += "\n--- STDERR ---:\n" + error_output
            else: # if only stderr, make it the main output
                output = "--- STDERR ---:\n" + error_output
        
        if len(output) > 5000:
            output = output[:5000] + "\n... (output truncated)"
        
        return output.strip() if output.strip() else "(empty output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except FileNotFoundError: # Specific error if the command itself is not found
        return f"Error: Command '{command.split()[0]}' not found. Please ensure it's installed and in PATH."
    except Exception as e:
        return f"Error executing command '{command}': {str(e)}"
