from crewai import Task
import re

def create_testing_task(agent_to_assign, file_to_test: str, test_requirements: str = None, context: str = None):
    """
    Creates a task for testing a specific file or component.

    Args:
        agent_to_assign: The QA agent.
        file_to_test (str): The path to the file that needs to be tested.
        test_requirements (str, optional): Specific requirements or functionalities to test for this file.
                                          If None, agent should infer from code or general best practices.
        context (str, optional): Additional context, e.g., output from a previous coding task.
    """
    
    # Generate a safe filename for the report
    safe_filename_base = re.sub(r'[^a-zA-Z0-9_.-]', '_', file_to_test)
    report_filename = f"TEST_REPORT_{safe_filename_base.replace('/', '_').replace('.', '_')}.md"

    description = (
        f"Thoroughly test the file located at '{file_to_test}'. "
        "Review its code for quality, potential bugs, and adherence to best practices. "
    )
    if test_requirements:
        description += f"Specific test requirements: {test_requirements}. "
    else:
        description += "Focus on general functionality, error handling, and common pitfalls. "
    
    description += (
        "If possible, write and execute unit tests for its functions/classes. "
        "Report all findings, including passed tests, failed tests, identified bugs, and areas of concern. "
        f"Save a summary of your findings and any test scripts created to a file named '{report_filename}'."
    )

    if context:
        description = f"Relevant Context from previous step:\n{context}\n\nTask:\n{description}"
        
    return Task(
        description=description,
        expected_output=(
            f"A comprehensive test report file named '{report_filename}'. "
            "This report should detail the testing process, executed tests, findings (bugs, issues, suggestions), "
            f"and overall quality assessment of the file '{file_to_test}'."
        ),
        agent=agent_to_assign
    )
