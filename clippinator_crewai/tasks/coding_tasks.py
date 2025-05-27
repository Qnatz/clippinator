from crewai import Task

# Note: The agent is typically assigned when the task is added to a crew,
# or if the task itself needs to specify a default agent.
# For now, we define the task, and it will be assigned to an agent (e.g., CodeWriterAgent)
# in the main crew script.

def create_hello_world_task(agent_to_assign=None):
    """
    Creates a task to write a 'hello.py' file with a simple 'Hello, World!' program.

    Args:
        agent_to_assign: The agent that will be assigned to perform this task.
                         This is typically set when the task is added to a Crew.
    
    Returns:
        A CrewAI Task instance.
    """
    return Task(
        description=(
            "Create a Python file named 'hello.py' in the current working directory. "
            "The file should contain a main block that, when executed, "
            "prints the string 'Hello, World!' to the console. "
            "Ensure the file has the correct Python syntax and is runnable."
        ),
        expected_output=(
            "A file named 'hello.py' created in the current working directory. "
            "The file content should be a Python script that, when run, prints 'Hello, World!' to standard output. "
            "For example:\n"
            "```python\n"
            "if __name__ == '__main__':\n"
            "    print('Hello, World!')\n"
            "```"
        ),
        agent=agent_to_assign, # Agent is assigned here or when creating the crew
        # human_input=False # Default is False, so not strictly necessary to state
    )

# Example of how this task might be instantiated and assigned to an agent
# (typically in a main crew script):
#
# from ..agents.coding_agents import create_code_writer_agent 
# # Assuming an LLM instance is available, e.g., from langchain_openai
# # from langchain_openai import ChatOpenAI
# # my_llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7) 
#
# # Create an instance of the agent that will perform the task
# # code_writer = create_code_writer_agent(llm_instance=my_llm)
#
# # Create the task and assign the agent to it
# # hello_world_task = create_hello_world_task(agent_to_assign=code_writer)
#
# # This task would then be added to a Crew.
# # from crewai import Crew
# # my_crew = Crew(agents=[code_writer], tasks=[hello_world_task], ...)
# # result = my_crew.kickoff()

def create_coding_task(filename: str, description: str, agent_to_assign, context: str = None):
    """
    Creates a generic coding task.

    Args:
        filename (str): The name of the file to be created/modified.
        description (str): Detailed description of the code to be written or changes to be made.
                           This will typically come from an architect agent.
        agent_to_assign: The agent responsible for this task.
        context (str, optional): Additional context for the task, e.g., relevant parts of an architecture document.
    
    Returns:
        A CrewAI Task instance.
    """
    
    task_full_description = (
        f"Your task is to write or modify the file named '{filename}'.\n"
        f"Detailed requirements: {description}\n"
        "Ensure your code is clean, well-commented, and adheres to best practices. "
        "Use the available file writing tools to save your changes to the specified filename."
    )
    if context:
        task_full_description = f"Relevant Context:\n{context}\n\nTask:\n{task_full_description}"

    return Task(
        description=task_full_description,
        expected_output=(
            f"The file '{filename}' successfully created or modified according to the provided description. "
            "The code should be functional and well-documented."
        ),
        agent=agent_to_assign
    )
