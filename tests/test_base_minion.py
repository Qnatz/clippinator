import pytest
from unittest.mock import Mock, call # Use call to check call arguments if needed

# Import the function to be tested
from clippinator.minions.base_minion import run_with_retries

# 1. Test successful execution (no retry)
def test_run_with_retries_successful_first_try():
    mock_agent_executor = Mock()
    mock_agent_executor.invoke.return_value = "success"
    inputs = {"key": "value"}
    
    result = run_with_retries(mock_agent_executor, inputs, max_retries=3)
    
    mock_agent_executor.invoke.assert_called_once_with(inputs)
    assert result == "success"

# 2. Test retry on "Invalid Format" exception
def test_run_with_retries_invalid_format_retry():
    mock_agent_executor = Mock()
    mock_agent_executor.invoke.side_effect = [
        Exception("Invalid Format: details"),
        "success_after_retry"
    ]
    inputs = {"initial_scratchpad": "some_initial_content"}
    
    result = run_with_retries(mock_agent_executor, inputs, max_retries=3)
    
    assert mock_agent_executor.invoke.call_count == 2
    # Check that the inputs dict was modified for the second call
    expected_reminder = "\n\nREMINDER: You MUST follow the Thought -> Action -> Action Input format. Every Thought needs an Action!"
    # The first call gets original inputs, second call gets modified inputs
    first_call_args, _ = mock_agent_executor.invoke.call_args_list[0]
    second_call_args, _ = mock_agent_executor.invoke.call_args_list[1]
    
    assert first_call_args[0]["initial_scratchpad"] == "some_initial_content" # original scratchpad
    assert "agent_scratchpad" not in first_call_args[0] # if it wasn't there initially

    assert second_call_args[0]["initial_scratchpad"] == "some_initial_content" # original key still there
    assert second_call_args[0]["agent_scratchpad"] == "some_initial_content" + expected_reminder if "some_initial_content" else expected_reminder


    assert result == "success_after_retry"

# 3. Test retry on "timeout" exception
def test_run_with_retries_timeout_retry():
    mock_agent_executor = Mock()
    mock_agent_executor.invoke.side_effect = [
        Exception("some timeout error"),
        "success_after_timeout_retry"
    ]
    inputs = {"key": "value"}
    
    result = run_with_retries(mock_agent_executor, inputs, max_retries=3)
    
    assert mock_agent_executor.invoke.call_count == 2
    assert result == "success_after_timeout_retry"

# 4. Test other generic exceptions leading to retry
def test_run_with_retries_generic_exception_retry():
    mock_agent_executor = Mock()
    mock_agent_executor.invoke.side_effect = [
        Exception("Some other error"),
        "success_after_generic_retry"
    ]
    inputs = {"key": "value"}
    
    result = run_with_retries(mock_agent_executor, inputs, max_retries=3)
    
    assert mock_agent_executor.invoke.call_count == 2
    assert result == "success_after_generic_retry"

# 5. Test exception re-raised after max retries
def test_run_with_retries_exception_after_max_retries():
    mock_agent_executor = Mock()
    mock_agent_executor.invoke.side_effect = Exception("Persistent error")
    inputs = {"key": "value"}
    
    with pytest.raises(Exception, match="Persistent error"):
        run_with_retries(mock_agent_executor, inputs, max_retries=3)
    
    assert mock_agent_executor.invoke.call_count == 3

# 6. Test `agent_scratchpad` initialization if not present for "Invalid Format"
def test_run_with_retries_invalid_format_scratchpad_initialization():
    mock_agent_executor = Mock()
    mock_agent_executor.invoke.side_effect = [
        Exception("Invalid Format: details"),
        "success_after_retry_no_initial_scratchpad"
    ]
    inputs = {} # No agent_scratchpad initially
    
    result = run_with_retries(mock_agent_executor, inputs, max_retries=3)
    
    assert mock_agent_executor.invoke.call_count == 2
    
    # The inputs dict for the second call should have agent_scratchpad initialized
    _, second_call_kwargs = mock_agent_executor.invoke.call_args_list[1] # Corrected to get kwargs if inputs is passed as kwargs
    # Or if inputs is passed as a positional argument:
    second_call_args_tuple = mock_agent_executor.invoke.call_args_list[1][0]
    modified_inputs_for_second_call = second_call_args_tuple[0]


    expected_reminder = "\n\nREMINDER: You MUST follow the Thought -> Action -> Action Input format. Every Thought needs an Action!"
    assert modified_inputs_for_second_call.get("agent_scratchpad") == expected_reminder
    assert result == "success_after_retry_no_initial_scratchpad"

# Additional test for "Invalid Format" when agent_scratchpad exists but is empty
def test_run_with_retries_invalid_format_empty_scratchpad():
    mock_agent_executor = Mock()
    mock_agent_executor.invoke.side_effect = [
        Exception("Invalid Format: details"),
        "success_after_retry_empty_scratchpad"
    ]
    inputs = {"agent_scratchpad": ""} # agent_scratchpad is present but empty
    
    result = run_with_retries(mock_agent_executor, inputs, max_retries=3)
    
    assert mock_agent_executor.invoke.call_count == 2
    
    second_call_args_tuple = mock_agent_executor.invoke.call_args_list[1][0]
    modified_inputs_for_second_call = second_call_args_tuple[0]
    
    expected_reminder = "\n\nREMINDER: You MUST follow the Thought -> Action -> Action Input format. Every Thought needs an Action!"
    assert modified_inputs_for_second_call["agent_scratchpad"] == "" + expected_reminder
    assert result == "success_after_retry_empty_scratchpad"

# Additional test for "Invalid Format" when agent_scratchpad exists and has content
def test_run_with_retries_invalid_format_existing_scratchpad_content():
    mock_agent_executor = Mock()
    mock_agent_executor.invoke.side_effect = [
        Exception("Invalid Format: details"),
        "success_after_retry_existing_content"
    ]
    initial_content = "This is some existing content."
    inputs = {"agent_scratchpad": initial_content}
    
    result = run_with_retries(mock_agent_executor, inputs, max_retries=3)
    
    assert mock_agent_executor.invoke.call_count == 2
    
    second_call_args_tuple = mock_agent_executor.invoke.call_args_list[1][0]
    modified_inputs_for_second_call = second_call_args_tuple[0]

    expected_reminder = "\n\nREMINDER: You MUST follow the Thought -> Action -> Action Input format. Every Thought needs an Action!"
    assert modified_inputs_for_second_call["agent_scratchpad"] == initial_content + expected_reminder
    assert result == "success_after_retry_existing_content"

# Test for inputs not being modified in non-Invalid Format errors
def test_inputs_not_modified_for_non_invalid_format_errors():
    mock_agent_executor = Mock()
    mock_agent_executor.invoke.side_effect = [
        Exception("some timeout error"),
        "success_after_timeout_retry"
    ]
    inputs = {"key": "value", "agent_scratchpad": "original_scratch"} # include agent_scratchpad
    original_inputs_copy = inputs.copy()

    run_with_retries(mock_agent_executor, inputs, max_retries=3)

    assert mock_agent_executor.invoke.call_count == 2
    
    # Check inputs for the second call
    second_call_args_tuple = mock_agent_executor.invoke.call_args_list[1][0]
    modified_inputs_for_second_call = second_call_args_tuple[0]

    # inputs should be the same as original_inputs_copy for non-Invalid Format errors
    assert modified_inputs_for_second_call == original_inputs_copy
    # Specifically, agent_scratchpad should not have the reminder
    assert "REMINDER:" not in modified_inputs_for_second_call.get("agent_scratchpad", "")

# Test case for max_retries = 1 (should try once, then raise)
def test_run_with_retries_max_retries_one():
    mock_agent_executor = Mock()
    mock_agent_executor.invoke.side_effect = Exception("Persistent error")
    inputs = {"key": "value"}

    with pytest.raises(Exception, match="Persistent error"):
        run_with_retries(mock_agent_executor, inputs, max_retries=1)

    assert mock_agent_executor.invoke.call_count == 1
