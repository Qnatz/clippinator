<img src="clippy.jpg" width="300"></img>

# Clippinator


[![GitHub Repo stars](https://img.shields.io/github/stars/ennucore/clippinator?style=social)](https://github.com/ennucore/clippinator)
[![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/ennucore/clippinator)](https://github.com/ennucore/clippinator/issues)
[![Twitter Follow](https://img.shields.io/twitter/follow/ennucore?style=social)](https://twitter.com/ennucore)

_A code assistant_

_(Formerly known as Clippy)_

[Twitter thread](https://twitter.com/ennucore/status/1680971027931693063)

### Core Change: Local LLM Powered

Clippinator now exclusively uses a local GGUF-compatible LLM (e.g., Deepseek Coder) via `llama-cpp-python`. This means it no longer relies on external API providers like OpenAI for its core agent functionalities.

### Getting started

1.  **Install Poetry:** Follow the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).
2.  **Clone this repository:** `git clone https://github.com/ennucore/clippinator.git && cd clippinator`
3.  **Obtain a GGUF Model:** Download a GGUF-format model file compatible with `llama-cpp-python`. Models like Deepseek Coder (e.g., `deepseek-coder-1.3b-instruct.Q4_K_M.gguf`) are good choices. Create a directory (e.g., `./models`) and place your GGUF file there.
4.  **Install Build Tools & Dependencies for `llama-cpp-python`:**
    *   `llama-cpp-python` compiles C++ code. You'll likely need build tools:
        *   On Debian/Ubuntu: `sudo apt-get install build-essential cmake`
        *   On macOS: Install Xcode Command Line Tools.
    *   For potential GPU acceleration (NVIDIA GPUs), ensure you have the CUDA toolkit installed. Refer to `llama-cpp-python` documentation for specifics.
    *   *(Termux users, see "Termux Installation Notes" below for specific dependencies.)*
5.  **Install `llama-cpp-python` and other dependencies:**
    *   It's recommended to install `llama-cpp-python` in a way that matches your hardware capabilities (e.g., with OpenBLAS, cuBLAS, Metal support). For example:
        ```bash
        # Example for OpenBLAS (CPU optimization)
        # CMAKE_ARGS="-DLLAMA_OPENBLAS=ON" pip install llama-cpp-python
        # Example for NVIDIA GPU (cuBLAS)
        # CMAKE_ARGS="-DLLAMA_CUBLAS=ON" pip install llama-cpp-python
        # Example for Apple Metal (M-series Macs)
        # CMAKE_ARGS="-DLLAMA_METAL=ON" pip install llama-cpp-python
        ```
        Refer to the [llama-cpp-python documentation](https://github.com/abetlen/llama-cpp-python) for detailed installation instructions tailored to your system.
    *   After potentially installing `llama-cpp-python` with specific CMAKE_ARGS, install all other project dependencies using Poetry:
        ```bash
        poetry install
        ```
        *Note: If `llama-cpp-python` is already listed in `pyproject.toml` (which it should be after this refactor, though not explicitly added by this subtask), `poetry install` might attempt to build it. Ensure your environment is prepared *before* running `poetry install` if it handles the `llama-cpp-python` build directly.*
6.  **Configure Environment Variables:**
    *   Copy the `.env.example` file to `.env`: `cp .env.example .env`
    *   Edit `.env` to set the required LlamaCpp parameters, primarily `MODEL_PATH`. Example:
        ```
        MODEL_PATH=./models/deepseek-coder-1.3b-instruct.Q4_K_M.gguf
        N_GPU_LAYERS=0 # Adjust if you have GPU support and want to offload layers
        N_CTX=2048
        # Other LlamaCpp parameters as needed (see .env.example)
        ```
    *   Optionally, set `SERPAPI_API_KEY` if you need search capabilities for certain tools.
7.  **Install `ctags`:** Follow the instructions on the [ctags website](https://docs.ctags.io/en/latest/building.html).
    *   For Termux users: `pkg install universal-ctags`
8.  **Install `pylint` (Optional but Recommended):** If `pylint` and `pylint-venv` are included as dependencies in `pyproject.toml`, `poetry install` will handle this. Otherwise, you might need to install them manually in your Python environment (e.g., `pip install pylint pylint-venv`).
9.  **Run Clippinator:**
    ```bash
    poetry run clippinator --help
    poetry run clippinator PROJECT_PATH
    ```
10. You can stop it (Ctrl+C) and then it will continue from the last saved state. Use Ctrl+C during agent operation to provide feedback to the main agent.

### Termux Installation Notes

Running Clippinator on Termux requires some specific steps due to its mobile environment:

1.  **Install Core Build Dependencies:**
    ```bash
    pkg update && pkg upgrade
    pkg install build-essential cmake rust libopenblas
    ```
    *   `rust` is needed for the `tiktoken` dependency.
    *   `libopenblas` can be used by `llama-cpp-python` for CPU acceleration.
2.  **Install `llama-cpp-python` on Termux:**
    *   You might need to specify `CMAKE_ARGS` to build `llama-cpp-python` correctly and enable features like OpenBLAS:
        ```bash
        CMAKE_ARGS="-DLLAMA_OPENBLAS=ON -DCMAKE_SYSTEM_NAME=Android -DCMAKE_SYSTEM_VERSION=$(getprop ro.build.version.sdk)" pip install llama-cpp-python
        ```
    *   Building `llama-cpp-python` can take a significant amount of time on device.
3.  **Install Project Dependencies:** After successfully installing `llama-cpp-python`, run:
    ```bash
    poetry install
    ```
4.  **Install `ctags`:** `pkg install universal-ctags`
5.  **Web Browsing Tool:** The built-in web browsing tool (`BrowseWebPage`) uses HTTP requests and does not require a separate browser installation or WebDriver setup on Termux.

### Performance & Configuration Notes

Several internal optimizations and configurations have been implemented, which can be particularly relevant for performance tuning, especially when using local LLMs or on resource-constrained devices:

*   **`ctags` Caching:** Summaries generated using `ctags` for project structure analysis are now cached. The cache is invalidated if a file's modification time changes, reducing redundant `ctags` executions for unchanged files.
*   **Optimized `pylint` Execution:** When linting multiple files or directories, `pylint` is now invoked once with all targets, rather than per file, making the process more efficient.
*   **Configurable Auto-Linting:** The `WriteFile` tool, used by agents to write files, has an internal `auto_lint_on_write` parameter (defaults to `True`). This controls whether Pylint is automatically run after a Python file is written. This is currently a code-level configuration.
*   **Configurable Summarization Context:** The context window for summarization (`max_context_length`) and the number of recent thoughts to retain before summarizing (`keep_n_last_thoughts`) are configurable parameters within the `CustomPromptTemplate` and `BaseMinion` classes. This allows for fine-tuning the balance between context detail and summarization frequency, which can impact performance and token usage. These are currently code-level configurations.
*   **Termux Compatibility:**
    *   The `PATH` environment variable handling in terminal tools has been revised to be safer for Termux, preventing the removal of essential system paths.
    *   Hardcoded `/bin/bash` paths have been changed to `bash` to rely on the system `PATH`.

These configurations are primarily intended for developers working with the Clippinator codebase. Future versions may expose some of these settings through more direct user interfaces or configuration files.

## Details

The purpose of **Clippinator** is to develop code for or with the user.
It can plan, write, debug, and test some projects autonomously.
For harder tasks, the best way to use it is to look at its work and provide feedback to it.

![](images/writing.png)
![](images/testing.png)

The tool consists of several agents that work together to help the user develop code. These agents are now powered by a local GGUF-compatible LLM (e.g., Deepseek Coder) run via `llama-cpp-python`.
This local operation means you are not reliant on external API providers or incurring per-token costs for the core LLM functionalities.

Here is the thing: it has a reasonable workflow by its own. It knows what to do and can do it. When it works, it works
faster than a human.
However, it's not perfect, and it can often make mistakes. But in combination with a human, it is very powerful.

Obviously, if you ask it to do something at very low levels of abstractions, like "Write a function that does X", it
will do it. It poses tasks like that to itself on its own, to a varying degree of success.
But combined with you, it will be able to do everything while only requiring a little bit of your intervention.
If the project is easy, you will just provide the most high-level guidance ("Write a link shortener web service"),
and if it's more complicated, you will be more involved, but **Clippinator** will still do most of the work.

![](images/map.png)

### Taskmaster

This tool has the main agent called _Taskmaster_. It is responsible for the overall development. It can use tools and
delegate tasks to subagents. To be able to run for a
long time, the history is summarized.

Taskmaster calls the specialized subagents (minions), like _Architect_ or _Writer_.

The taskmaster first asks some questions to the user to understand the project.
Then it asks the Architect to plan the project structure, and then it writes, debugs, and tests the project by
delegating tasks to the subagents.

### Minions

All agents have access to the planned project architecture, current project structure, errors from the linter, memory.
The agents use different tools, like writing to files, using bash (including running background commands), using the
browser with Selenium, etc.

We have the following agents: _Architect_, _Writer_, _Frontender_, _Editor_, _QA_, _Devops_. They all have different
prompts and
tools.

### Architecture

The architecture is just text which is written by the Architect.
It is a list of files with summaries of their contents in the form of comments, important lines (like classes and
functions).

![](images/architecture.png)

The architecture is made available to all agents. Implementing architecture is the goal of the agents at the first
stages.

### Tools

A variety of tools have been implemented (or taken from Langchain):

- File tools: WriteFile, ReadFile, other tools which aren't used at the moment.
- Terminal tools: RunBash, Python, BashBackground (allows to start and manage background processes like starting a
  server).
- Human input
- Pylint
- BrowseWebPage - Fetches text content from web pages and can follow links to a specified depth. Uses HTTP requests and HTML parsing (does not execute JavaScript).
- HttpGet, GetPage - simpler tools for getting a page (Note: GetPage might be considered for deprecation in favor of BrowseWebPage's core functionality if only text is needed)
- DeclareArchitecture, SetCI, Remember - allow the agents to set up their environment, write architecture, remember
  things

### Project structure, Linting, CI, Memory

![](images/structure.png)
One important part of it is the project structure, which is given to all agents.
It is a list of files with some of their important lines, obtained by ctags.

Linter output is given next to the project structure. It is very helpful to understand the current issues of the
project.
Linter output is also given after using _WriteFile_.
The architect can configure the linter command using the _SetCI_ tool.
All agents can also use the _Remember_ tool to add some information to the memory. Memory is given to all agents.

### Feedback

You can press ^C to provide feedback to the main agent. Note that if you press it during the execution of a subagent,
the subagent will be aborted. The only exception here is the _Architect_: you can press ^C after it uses
the `DeclareArchitecture` tool to ask it to change it.

After the architect is ran, you can also edit the project architecture manually if you choose `y` in the prompt.

If you enter `m` or `menu`, you will also be able to edit the project architecture, objective, and other things.


_Created by Lev Chizhov and Timofey Fedoseev with contributions by Sergei Bogdanov_

[Twitter thread](https://twitter.com/ennucore/status/1680971027931693063)
