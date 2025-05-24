import os
import sys
import pytest # Using pytest for structure, can be run with `pytest tests/`
import shutil # For cleaning up the dummy project

# Add clippinator to sys.path to allow imports from the parent directory
# This might be necessary depending on how tests are run.
# A better approach might be to install clippinator in editable mode (pip install -e .)
# For now, let's add a common way to handle imports in tests.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clippinator.minions.taskmaster import Taskmaster
from clippinator.project import Project

# A fixture to initialize Taskmaster might be useful if multiple tests need it.
# For now, initialize directly in the function.

def test_validate_agent(): # Renamed function
    """
    Runs a series of test cases against the agent_executor to validate
    basic functionality.
    """
    # Setup: Initialize a dummy project and Taskmaster
    # Ensure necessary environment variables for CustomLlamaCliLLM are set,
    # or mock/configure LLM for testing if actual LLM calls are too slow/costly.
    # For this task, assume environment variables are set for LLM.
    
    # Create a dummy project path if it doesn't exist for testing
    dummy_project_path = "test_dummy_project"
    if not os.path.exists(dummy_project_path):
        os.makedirs(dummy_project_path)
    
    # Basic project setup
    # NOTE: The actual 'objective' and 'architecture' might not be strictly necessary
    # for these specific test cases if they don't rely on complex project context.
    # However, Taskmaster expects a Project instance.
    try:
        project = Project(
            path=dummy_project_path,
            objective="Test project for validation checklist.",
            architecture="Test architecture: simple_app.py",
            current_milestone_plan="Test milestone: Create basic files."
        )
        
        # Ensure Taskmaster is initialized (which uses DebuggingAgentExecutor)
        # This assumes that CustomLlamaCliLLM can be initialized (env vars are set)
        taskmaster_instance = Taskmaster(project=project)
        agent_executor = taskmaster_instance.agent_executor 
        # If Taskmaster or CustomLlamaCliLLM init fails, this will raise an exception.

    except Exception as e:
        pytest.fail(f"Failed to initialize Taskmaster or Project for validation: {e}")
        return # Should not be reached if pytest.fail is used

    test_cases = [
        ("Create homepage HTML", "should generate HTML file"), 
        # The expectation "should generate HTML file" is vague for an automated assert.
        # The agent's actual output would be a string, often a 'Final Answer'.
        # We need to assert based on the 'output' of agent_executor.invoke.
        # Let's refine expectations to something more checkable in the output string.
        # For now, we'll assume the 'Final Answer' will contain certain keywords.
        ("Add CSS styling for the homepage", "should create CSS file"),
        ("Implement a responsive mobile menu using JavaScript", "should include JavaScript")
    ]
    
    results = []
    for prompt, expectation_keywords in test_cases:
        print(f"\nRunning validation for prompt: '{prompt}'")
        try:
            # The agent_executor now expects a dictionary with 'input' key.
            # The 'objective' from the issue's taskmaster_prompt is now the primary input.
            # So, we map the test prompt to the 'objective' (or 'input') field.
            # The Taskmaster's invoke method also maps 'objective' to 'input'.
            # The CustomPromptTemplate for Taskmaster expects 'objective', 'project_name', and 'history'.
            invoke_input = {
                "input": prompt,  # Primary input for agent execution
                "objective": prompt, # For the {objective} field in the prompt
                "project_name": project.name, # For the {project_name} field in the prompt
                "history": "" # For the {history} field in the prompt, initially empty
            }
            result = agent_executor.invoke(invoke_input)
            
            # The actual result is a dictionary. We're interested in 'output'.
            output_text = result.get('output', '')
            print(f"Agent output: {output_text[:300]}...") # Print a snippet of the output

            # Basic assertion: check if expectation keywords are in the output string
            # This is a weak assertion but matches the spirit of the issue's example.
            assert expectation_keywords.lower() in str(output_text).lower(), \
                   f"Failed for prompt: '{prompt}'. Expected keywords '{expectation_keywords}' not found in output: '{output_text}'"
            results.append(f"PASSED: {prompt}")
        except Exception as e:
            # If an assertion fails, it will be an AssertionError, handled by pytest.
            # Other exceptions during invoke should also fail the test.
            results.append(f"FAILED: {prompt} with exception: {e}")
            pytest.fail(f"Exception during agent invocation for prompt '{prompt}': {e}")

    print("\n=== VALIDATION SUMMARY ===")
    for res_summary in results:
        print(res_summary)
    
    # Clean up dummy project directory
    # Be careful with rmtree, ensure it's the correct path.
    if os.path.exists(dummy_project_path):
        shutil.rmtree(dummy_project_path)

# Add a way to run this validation if the file is executed directly
if __name__ == "__main__":
    # We need to run the specific function, not just the file with pytest.main
    # Pytest is good for test discovery, but for direct execution, just call the function.
    # However, to keep the pytest structure (like pytest.fail), we can still use pytest.main
    # to run this specific test function.
    # To run a specific function with pytest.main: pytest.main([__file__, "-k", "test_validate_agent"]) # Renamed here too
    # Or, more simply, if we want to run all tests in this file:
    pytest.main([__file__])
