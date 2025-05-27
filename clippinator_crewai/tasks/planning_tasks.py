from crewai import Task

def create_initial_planning_task(agent_to_assign, objective: str):
    return Task(
        description=(
            f"Analyze the following project objective: '{objective}'. "
            "If any part of the objective is unclear, you should try to make reasonable assumptions or state what needs clarification. "
            "Produce a detailed project plan with clear milestones and specific, actionable tasks for each milestone. "
            "The plan should be formatted in markdown. "
            "Save this comprehensive plan to a file named 'PROJECT_PLAN.md' in the current working directory using the available file writing tool."
        ),
        expected_output=(
            "A markdown file named 'PROJECT_PLAN.md' created in the current working directory. "
            "This file must contain a comprehensive project plan, including milestones and actionable tasks derived from the objective."
        ),
        agent=agent_to_assign
    )
