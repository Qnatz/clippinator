import unittest
from unittest.mock import patch, MagicMock

from langchain_core.tools import BaseTool # Import BaseTool

from clippinator.minions.taskmaster import Taskmaster
from clippinator.project.project import Project # Adjusted import
from clippinator.minions.base_minion import BasicLLM as ActualBasicLLM # Import actual BasicLLM for isinstance checks


class TestTaskmaster(unittest.TestCase):

    @patch('clippinator.minions.taskmaster.CustomLlamaCliLLM')
    @patch('clippinator.minions.taskmaster.get_specialized_executioners')
    @patch('clippinator.minions.taskmaster.Executioner')
    @patch('clippinator.minions.taskmaster.get_tools')
    @patch('clippinator.minions.taskmaster.SelfCall')
    @patch('clippinator.minions.taskmaster.create_react_agent')
    @patch('clippinator.minions.taskmaster.DebuggingAgentExecutor')
    @patch('clippinator.minions.taskmaster.BasicLLM')
    @patch('clippinator.minions.taskmaster.Subagent')
    @patch('clippinator.minions.taskmaster.WarningTool')
    @patch('clippinator.minions.taskmaster.CustomPromptTemplate') # Mock CustomPromptTemplate
    def test_taskmaster_init_with_config_defaults(
        self,
        MockCustomPromptTemplate, # Added CustomPromptTemplate mock
        MockWarningTool,
        MockSubagent,
        MockBasicLLM,
        MockDebuggingAgentExecutor,
        MockCreateReactAgent,
        MockSelfCall,
        MockGetTools,
        MockExecutioner,
        MockGetSpecializedExecutioners,
        MockCustomLlamaCliLLM
    ):
        # Arrange
        mock_llm_instance = MockCustomLlamaCliLLM.return_value
        
        # Basic mocking for tool providers, as CustomPromptTemplate is now mocked
        MockGetTools.return_value = [] 
        mock_self_call_instance = MockSelfCall.return_value
        mock_self_call_instance.get_tool.return_value = MagicMock()
        mock_subagent_instance = MockSubagent.return_value
        mock_subagent_instance.get_tool.return_value = MagicMock()
        mock_warning_tool_instance = MockWarningTool.return_value
        mock_warning_tool_instance.get_tool.return_value = MagicMock()
        MockBasicLLM.return_value = MagicMock() # Basic mock for BasicLLM

        project_path = "dummy_project_path"
        project_objective = "dummy_objective"

        # Test with empty config
        project_empty_config = Project(path=project_path, objective=project_objective, config={})
        Taskmaster(project=project_empty_config)
        MockCustomLlamaCliLLM.assert_called_with(
            cli_path='/default/path/to/llama-cli',
            model_path='/default/path/to/model.gguf',
            n_ctx=2048,
            n_threads=4
        )

        # Reset mock for next assertion
        MockCustomLlamaCliLLM.reset_mock()

        # Test with partial config
        project_partial_config = Project(
            path=project_path,
            objective=project_objective,
            config={"cli_path": "/custom/cli", "n_ctx": "1024"}
        )
        Taskmaster(project=project_partial_config)
        MockCustomLlamaCliLLM.assert_called_with(
            cli_path='/custom/cli',
            model_path='/default/path/to/model.gguf',
            n_ctx="1024", # Note: The issue description implies n_ctx can be a string
            n_threads=4
        )

        # Reset mock for next assertion
        MockCustomLlamaCliLLM.reset_mock()

        # Test with full config
        project_full_config = Project(
            path=project_path,
            objective=project_objective,
            config={
                "cli_path": "/another/custom/cli",
                "model_path": "/another/custom/model.gguf",
                "n_ctx": "4096",
                "n_threads": "8"
            }
        )
        Taskmaster(project=project_full_config)
        MockCustomLlamaCliLLM.assert_called_with(
            cli_path='/another/custom/cli',
            model_path='/another/custom/model.gguf',
            n_ctx="4096",
            n_threads="8" # Note: The issue description implies n_threads can be a string
        )

if __name__ == '__main__':
    unittest.main()
