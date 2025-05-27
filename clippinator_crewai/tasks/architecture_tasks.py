from crewai import Task

def create_architecture_design_task(agent_to_assign, context: str = None):
    # context would ideally be the content of PROJECT_PLAN.md or its path
    task_description = (
        "Read the project plan (expected to be found in 'PROJECT_PLAN.md' in the current working directory, "
        "or its content will be provided in the context). "
        "Based on this plan, design a comprehensive software architecture. "
        "The architecture should detail the file structure, main components, classes, functions, "
        "and their interactions. Consider the technology stack if specified or make sensible recommendations. "
        "Document this architecture clearly in markdown format and save it to a file named 'PROJECT_ARCHITECTURE.md' "
        "in the current working directory using the available file writing tool."
    )
    if context:
        task_description = f"Context: {context}\n\nTask: {task_description}"
        
    return Task(
        description=task_description,
        expected_output=(
            "A markdown file named 'PROJECT_ARCHITECTURE.md' created in the current working directory. "
            "This file must contain a detailed software architecture design, including file structures, "
            "key components, and their interactions, based on the provided project plan."
        ),
        agent=agent_to_assign,
        # If this task depends on the output of the planning task (e.g., the content of PROJECT_PLAN.md),
        # that dependency would be set up in the Crew definition using the `context` parameter of the Task
        # or by ensuring the planner task makes its output available in a way this task can access.
        # For now, the description guides the agent to look for 'PROJECT_PLAN.md'.
    )
