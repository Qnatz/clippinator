import pytest
from clippinator.project.project import Project

# Test case 1: Dictionary input with all keys
def test_update_state_dict_all_keys():
    project = Project(path="./dummy_path", objective="test_objective")
    initial_memories = project.memories.copy() # Ensure we have a copy if project.memories can be None initially
    if initial_memories is None:
        initial_memories = []

    update_data = {'state': 'new_state', 'architecture': 'new_arch', 'memories': ['mem1']}
    project.update_state(update_data)
    
    assert project.state == 'new_state'
    assert project.architecture == 'new_arch'
    assert project.memories == initial_memories + ['mem1']

# Test case 2: Dictionary input with some keys missing
def test_update_state_dict_some_keys_missing():
    project = Project(path="./dummy_path", objective="test_objective", state="initial_state", architecture="initial_arch", memories=['initial_mem'])
    original_architecture = project.architecture
    original_memories = project.memories.copy()

    update_data = {'state': 'another_state'}
    project.update_state(update_data)
    
    assert project.state == 'another_state'
    assert project.architecture == original_architecture  # Should remain unchanged
    assert project.memories == original_memories  # Should remain unchanged

# Test case 3: Dictionary input with memories extension
def test_update_state_dict_memories_extension():
    project = Project(path="./dummy_path", objective="test_objective", memories=['initial_mem'])
    
    update_data = {'memories': ['new_mem1', 'new_mem2']}
    project.update_state(update_data)
    
    assert project.memories == ['initial_mem', 'new_mem1', 'new_mem2']

# Test case 4: Plain string input
def test_update_state_plain_string():
    project = Project(path="./dummy_path", objective="test_objective", architecture="initial_arch", memories=['initial_mem'])
    original_architecture = project.architecture
    original_memories = project.memories.copy()

    update_data = "plain string state"
    project.update_state(update_data)
    
    assert project.state == "plain string state"
    assert project.architecture == original_architecture  # Should remain unchanged
    assert project.memories == original_memories  # Should remain unchanged

# Test case 5: Non-string/non-dict input (e.g., integer)
def test_update_state_non_string_non_dict():
    project = Project(path="./dummy_path", objective="test_objective", architecture="initial_arch", memories=['initial_mem'])
    original_architecture = project.architecture
    original_memories = project.memories.copy()
    
    update_data = 123
    project.update_state(update_data)
    
    assert project.state == '123'  # Should be converted to string
    assert project.architecture == original_architecture  # Should remain unchanged
    assert project.memories == original_memories  # Should remain unchanged

# Test case for dictionary input with only architecture
def test_update_state_dict_only_architecture():
    project = Project(path="./dummy_path", objective="test_objective", state="initial_state", memories=['initial_mem'])
    original_state = project.state
    original_memories = project.memories.copy()

    update_data = {'architecture': 'updated_arch'}
    project.update_state(update_data)

    assert project.state == original_state # Should remain unchanged
    assert project.architecture == 'updated_arch'
    assert project.memories == original_memories # Should remain unchanged

# Test case for empty dictionary input
def test_update_state_empty_dict():
    project = Project(path="./dummy_path", objective="test_objective", state="initial_state", architecture="initial_arch", memories=['initial_mem'])
    original_state = project.state
    original_architecture = project.architecture
    original_memories = project.memories.copy()

    update_data = {}
    project.update_state(update_data)

    assert project.state == original_state
    assert project.architecture == original_architecture
    assert project.memories == original_memories
