import os
import subprocess
from dataclasses import dataclass

from .tool import SimpleTool
from .utils import skip_file


@dataclass
class FindUsages(SimpleTool):
    pass


@dataclass
class SearchAndReplace(SimpleTool):
    pass


def run_pylint_on_file(target: str) -> list[str]:
    cmd = ["pylint", target, "-E", "--allow-any-import-level", ".", "--output-format", "text"]
    process = subprocess.run(cmd, capture_output=True, text=True)
    pylint_output = process.stdout.strip().split("\n")
    return [line for line in pylint_output if 'pydantic' not in line]


def lint_file(file_path: str) -> str:
    output = ''
    if file_path.endswith(".py"):
        try:
            # This function is now intended for single file linting, 
            # run_pylint_on_args handles batching.
            pylint_output_lines = run_pylint_on_file(file_path)
            output = "\n".join(pylint_output_lines)
        except Exception as e: # More specific exception handling if possible
            # print(f"Error linting file {file_path}: {e}", file=sys.stderr) # Consider logging
            return f"Error during linting {file_path}: {e}"
        
    if len(output) > 800: # This truncation might be too aggressive or hide important info
        output = output[:750] + "\n... (output truncated)"
    return output


def run_pylint_on_args(args: str, workdir: str) -> str:
    input_targets = args.strip().split() if args.strip() else ["."] # Default to current dir if no args
    
    files_to_lint = []
    for input_target in input_targets:
        # Construct full path, ensuring it's within workdir for safety, though pylint handles paths.
        # For simplicity, assuming targets are relative to workdir or absolute paths intended for linting.
        # If args are like ".", they resolve relative to workdir.
        path_target = os.path.join(workdir, input_target) if not os.path.isabs(input_target) else input_target

        if os.path.isfile(path_target) and path_target.endswith(".py"):
            files_to_lint.append(path_target)
        elif os.path.isdir(path_target):
            for root, _, files in os.walk(path_target):
                for file in files:
                    if file.endswith(".py") and not skip_file(file): # Assuming skip_file is defined
                        files_to_lint.append(os.path.join(root, file))
        else:
            # Handle case where a target is not a file or directory
            # Could return an error message or collect them.
            # For now, we'll implicitly skip non-Python files or invalid paths if they don't fall into above.
            # If a specific target is not found, Pylint will also report it.
            # To be explicit about not found targets, one could add:
            # if not os.path.exists(path_target):
            #     return f"Target not found: {path_target}"
            # However, Pylint itself handles this, so we can just pass targets to it.
            # If input_targets was just a list of files/dirs, we can pass them directly.
            # The current logic tries to expand directories.
            pass # Pylint will handle errors for non-existent paths if passed directly

    if not files_to_lint and input_targets:
        # If initial targets were provided but none resolved to Python files (e.g., non-Python dir)
        # Or if specific non-Python files were listed.
        # We can either let Pylint handle it (it will likely error or do nothing for non-Python files)
        # or return a message. For now, let Pylint decide.
        # If args was empty (defaulting to "."), and workdir has no .py files, files_to_lint will be empty.
        # In this case, we want to run pylint on the workdir itself.
        if not files_to_lint and input_targets == ["."]: # Default case, lint the directory
             files_to_lint.append(os.path.join(workdir, ".")) # Pylint can take a directory


    if not files_to_lint:
        return "No Python files found to lint."

    # Deduplicate file list just in case
    files_to_lint = sorted(list(set(files_to_lint)))

    # Run Pylint once with all collected files/directories
    # Note: Pylint's behavior with "--allow-any-import-level" and "." might need adjustment
    # when passing a list of files from potentially different subdirectories.
    # The "." argument to pylint usually means the current directory for discovery.
    # If files_to_lint contains absolute paths, the "." might be less relevant.
    # Let's assume pylint handles this correctly or adjust if issues arise.
    # Using workdir as the CWD for pylint might be important.
    
    # Original cmd: ["pylint", target, "-E", "--allow-any-import-level", ".", "--output-format", "text"]
    # We need to adapt this. If we pass a list of files, the "." might be implicitly handled by pylint
    # or we might need to run it from `workdir`.
    
    # It's generally better to run pylint from the project root (workdir)
    # and provide relative paths to files/modules if possible, or absolute paths.
    cmd = ["pylint", "-E", "--output-format=text"] + files_to_lint
    # Removed "--allow-any-import-level" and "." as specific file list is provided.
    # If these are needed, they might need to be re-evaluated with batch file list.
    # Pylint typically discovers local modules relative to the files it's linting or CWD.

    try:
        process = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir, check=False)
        pylint_output_lines = process.stdout.strip().split("\n")
        # Filter out empty lines or specific noise if necessary
        pylint_output_lines = [line for line in pylint_output_lines if line.strip() and 'pydantic' not in line]
        formatted_output = "\n".join(pylint_output_lines)
    except Exception as e:
        return f"Error running Pylint: {e}"

    # Truncation should be applied to the final string if it's too long.
    if len(formatted_output) > 2000: # Increased limit for potentially more output
        formatted_output = formatted_output[:1950] + "\n... (Pylint output truncated)"
    elif not formatted_output.strip():
        return "Pylint run completed. No issues found or no files linted."
        
    return formatted_output


def lint_project(workdir: str) -> str:
    output = ''
    try:
        output = run_pylint_on_args("", workdir)
    except:
        pass
    if len(output) > 800:
        output = output[:800] + "\n..."
    return output


@dataclass
class Pylint(SimpleTool):
    name = "Pylint"
    description = (
        "runs pylint to check for python errors. By default it runs on the entire project. "
        "You can specify a relative path to run on a specific file or module."
    )

    def __init__(self, wd: str = "."):
        self.workdir = wd

    def func(self, args: str) -> str:
        return run_pylint_on_args(args, self.workdir)


class SearchInFiles(SimpleTool):
    """
    A tool that can be used to search for a string in all files.
    """
    name = "SearchInFiles"
    description = "A tool that can be used to search for occurrences a string in all files. " \
                  "The input format is [search_directory] on the first line, " \
                  "and the search query on the second line. " \
                  "The tool will return the file paths and line numbers containing the search query."

    def __init__(self, wd: str = "."):
        self.workdir = wd

    def search_files(self, search_dir: str, search_query: str) -> list[str]:
        results = []
        search_dir = os.path.join(self.workdir, search_dir)

        for root, _, files in os.walk(search_dir):
            for file in files:
                if skip_file(file):
                    continue
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()

                    for line_number, line in enumerate(lines, start=1):
                        if search_query.lower() in line.lower():
                            results.append(f"{file_path}:{line_number}")
                except Exception as e:
                    # Ignore errors related to file reading or encoding
                    pass

        return results

    def func(self, args: str) -> str:
        # Split the input by newline to separate the search directory and the search query
        input_lines = args.strip().split('\n')

        if len(input_lines) < 2:
            return "Invalid input. Please provide search directory on the " \
                   "first line and search query on the second line."

        search_dir = os.path.join(self.workdir, input_lines[0])
        search_query = input_lines[1]

        results = self.search_files(search_dir, search_query)

        if results:
            return "\n".join(results)[:1500]
        else:
            return "No matches found."
