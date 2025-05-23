import os

from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain_community.utilities import SerpAPIWrapper # Updated import

from clippinator.project import Project
from .architectural import Remember, TemplateInfo, TemplateSetup, SetCI, DeclareArchitecture
# Removed SeleniumTool, GetPage from .browsing
from .simple_web_browser import SimpleWebBrowserTool # Added SimpleWebBrowserTool
from .code_tools import SearchInFiles, Pylint
from .file_tools import WriteFile, ReadFile, PatchFile, SummarizeFile
from .terminal import RunBash, BashBackgroundSessions, RunPython
from .tool import HumanInputTool, HTTPGetTool, SimpleTool

tool_cache = {}


def fixed_tools(project: Project) -> list[SimpleTool]:
    if project.path in tool_cache:
        return tool_cache[project.path]
    result = [
        ReadFile(project.path),
        PatchFile(project.path),
        SummarizeFile(project.path),
        HumanInputTool(),
        Pylint(project.path),
        # SeleniumTool(), # Removed
        HTTPGetTool(),
        # GetPage(), # Removed
        TemplateInfo(),
        TemplateSetup(project),
        # Note: SimpleWebBrowserTool is not added here as a SimpleTool, 
        # it will be wrapped as a Langchain Tool directly in get_tools for better argument handling.
    ]
    tool_cache[project.path] = result
    return result


def get_tools(project: Project, try_structured: bool = False) -> list[BaseTool]:
    tools = [

                Tool(
                    name="Bash",
                    func=RunBash(workdir=project.path).run,
                    description="allows you to run bash commands in the project directory. "
                                "The input must be a valid bash command that will not ask for input and will terminate.",
                ),
                Tool(
                    name="Python",
                    func=RunPython(workdir=project.path).run,
                    description="allows you to run python code and get everything that's "
                                "printed (e.g. print(2+2) will give you 4) in order to compute something. "
                                "The input is correct python code.",
                ),
                # Tool(
                #     name="Wolfram Alpha",
                #     func=WolframAlphaAPIWrapper().run,
                #     description="allows you to ask questions about math, science, solve equations, and more. "
                #                 "The question should be strictly defined, like 'what is the derivative of x^2' or "
                #                 "'what is the capital of France'",
                # ),
                # Tool(
                #     name="Wait",
                #     func=lambda t: time.sleep(float(t)) or "",
                #     description="allows you to wait for a certain amount of time "
                #     "- to wait for the result of some process you ran.",
                # ),

                WriteFile(project).get_tool(try_structured),
                Remember(project).get_tool(try_structured),
                SetCI(project).get_tool(try_structured),
                # SearchInFiles(project.path).get_tool(),
                BashBackgroundSessions(project.path).get_tool(try_structured),
                DeclareArchitecture(project).get_tool(try_structured),
            ] + [tool_.get_tool(try_structured) for tool_ in fixed_tools(project)]
    
    # Add SimpleWebBrowserTool
    simple_browser_instance = SimpleWebBrowserTool()
    tools.append(
        Tool(
            name="BrowseWebPage",
            func=lambda q: simple_browser_instance.run(
                url=q.split(',')[0].strip(), 
                depth=int(q.split(',')[1].strip()) if ',' in q and q.split(',')[1].strip().isdigit() else 0,
                max_pages=int(q.split(',')[2].strip()) if q.count(',') >= 2 and q.split(',')[2].strip().isdigit() else 5,
                max_total_text_len=int(q.split(',')[3].strip()) if q.count(',') >= 3 and q.split(',')[3].strip().isdigit() else 10000
            ),
            description="Fetches text content of a web page and optionally crawls links. "
                        "Input should be a URL. Optional: add ', <depth>' (e.g., 'http://example.com, 1') to crawl. "
                        "Further optional: ', <max_pages>' (e.g., 'http://example.com, 1, 10'). "
                        "Further optional: ', <max_total_text_len>' (e.g., 'http://example.com, 1, 10, 15000'). "
                        "Defaults: depth=0, max_pages=5, max_total_text_len=10000. "
                        "Returns a dictionary with 'text', 'links', and 'visited_urls'.",
        )
    )
    
    if os.environ.get('SERPAPI_API_KEY'):
        tools.append(Tool(
            name="Search",
            func=SerpAPIWrapper().run,
            description="useful for when you need to answer simple questions and get a simple answer. "
                        "You cannot read websites or click on any links or read any articles.",
        ))
    return tools
