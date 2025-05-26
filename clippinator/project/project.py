from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from clippinator.project.project_summary import get_file_summary

@dataclass
class Project:
    path: str
    objective: str
    state: str = ""
    architecture: str = ""
    config: Dict[str, str] = field(default_factory=dict)
    summary_cache: str = ""
    template: str = "General"
    ci_commands: Dict[str, str] = field(default_factory=dict)
    memories: List[str] = field(default_factory=list)

    def get_history(self) -> str:
        """Format project memories into a historical context string"""
        if not self.memories:
            return "No project history yet"
        return "\n".join([f"- {memory}" for memory in self.memories])

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def get_folder_summary(self, path: str, indent: str = "", 
                          add_linting: bool = True, top_level: bool = False,
                          length_3: int = 20000) -> str:
        from clippinator.tools.utils import skip_file, skip_file_summary, trim_extra

        res = ""
        if not os.path.isdir(path):
            return ""
        
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if skip_file(file_path):
                continue
                
            if os.path.isdir(file_path):
                res += f"{indent}{file}:\n"
                res += self.get_folder_summary(file_path, indent + "  ", False)
            else:
                res += f"{indent}{file}\n"
                if not skip_file_summary(file_path):
                    res += get_file_summary(
                        file_path, 
                        indent + "  ",
                        length_1=length_3 // 11, 
                        length_2=round(length_3 / 7)
                        )
        if len(res) > length_3:
            print(f"Warning: long project summary at {path}, truncating to {length_3} chars")
            res = trim_extra(res, length_3)
            
        if not res.replace('-', '').strip() and top_level:
            return "(nothing in the project directory)"
            
        if add_linting:
            res += '\n--\n'
            res += self.lint(path)
            res += '\n-----\n'
            
        return res

    def lint(self, path: str = '') -> str:
        from clippinator.tools.code_tools import lint_project
        from clippinator.tools.utils import trim_extra

        path = os.path.join(self.path, path)
        path = path or self.path
        
        if self.ci_commands.get('lint'):
            cmd = self.ci_commands['lint']
            try:
                process = subprocess.run(
                    ['/bin/bash', '-c', cmd], 
                    capture_output=True,
                    text=True, 
                    cwd=self.path
                )
                return trim_extra(
                    process.stdout.strip() + process.stderr.strip(), 
                    3000, 
                    end_length=1500
                )
            except Exception as e:
                return f"Linter error: {e}"
                
        return lint_project(path)

    def lint_file(self, path: str) -> str:
        from clippinator.tools.code_tools import lint_file
        from clippinator.tools.utils import trim_extra

        path = os.path.join(self.path, path)
        
        if self.ci_commands.get('lintfile', '').strip():
            cmd = self.ci_commands['lintfile'] + ' ' + path
            try:
                process = subprocess.run(
                    ['/bin/bash', '-c', cmd], 
                    capture_output=True,
                    text=True, 
                    cwd=self.path
                )
                return trim_extra(process.stdout.strip(), 1000)
            except Exception as e:
                return f"Linter error: {e}"
                
        return lint_file(path)

    def get_project_summary(self) -> str:
        self.summary_cache = self.get_folder_summary(
            self.path, 
            top_level=True
        )
        return self.summary_cache
    
    def get_architecture_example(self) -> str:
        from clippinator.tools.architectural import templates
        return templates.get(
            self.template, {}
        ).get(
            'architecture', 
            templates['General']['architecture']
        )

    def update_state(self, new_state):
        """Update the project state with new information"""
        if isinstance(new_state, dict):
            # Extract relevant information from result
            if 'state' in new_state:
                self.state = new_state['state']
            if 'architecture' in new_state:
                self.architecture = new_state['architecture']
            if 'memories' in new_state:
                self.memories.extend(new_state.get('memories', []))
        else:
            # Handle string state updates
            self.state = str(new_state)
           
    def menu(self, prompt=None) -> None:
        from clippinator.tools.utils import select, get_input_from_editor
        
        options = ["Continue", "Architecture", "Objective", "Memories", "CI"]
        if prompt is not None:
            options.append("Edit action summary")
            
        res = select(options, "Project Menu")
        
        # Handle cancellation case
        if res < 0:
            return
        
        # Use 0-based indexing for options
        if res == 0:  # Continue
            return
        elif res == 1:  # Architecture
            self.architecture = get_input_from_editor(self.architecture)
        elif res == 2:  # Objective
            self.objective = get_input_from_editor(self.objective)
        elif res == 3:  # Memories
            self.memories = get_input_from_editor("\n".join(self.memories)).splitlines()
        elif res == 4:  # CI
            ci_commands = get_input_from_editor("\n".join(
                [f"{k}: `{v}`" for k, v in self.ci_commands.items()]
            )).splitlines()
            self.ci_commands = {}
            for line in ci_commands:
                if ':' in line:
                    key, value = line.split(':', 1)
                    self.ci_commands[key.strip()] = value.strip().strip('`')
        elif res == 5 and prompt is not None:  # Edit action summary
            prompt.last_summary = get_input_from_editor(prompt.last_summary)
    
    def prompt_fields(self) -> Dict[str, str]:
        return {
            "objective": self.objective,
            "project_name": self.name,
            "project_summary": self.get_project_summary(),
            "architecture": self.architecture,
            "history": self.get_history(),
            # Ensure we include any other variables used in the template
            "memories": "\n".join([f"- {m}" for m in self.memories]) if self.memories else "None",
            "architecture_example": self.get_architecture_example(),
            # "state" is still a field in Project, but if it's not explicitly used
            # as an input_variable in CustomPromptTemplate, it doesn't strictly need
            # to be here for the prompt formatting, but it doesn't hurt.
             "state": self.state, 
        }

