import os

# @tool("Writes content to a specified file. Input should be a string with filename and content separated by a newline, e.g., 'my_file.txt\\nFile content here'. Alternatively, it can be a dictionary with 'filename' and 'content' keys, though direct string input is also handled for simpler LLM interaction.")
def write_file_tool(filename: str, content: str = None) -> str:
    """
    Writes content to a specified file.
    Inputs:
        filename (str): The name of the file to write to. If content is not provided
                        separately, this string can contain the filename and content
                        separated by a newline.
        content (str, optional): The content to write to the file. If None, it's
                                 assumed filename contains both.
    """
    actual_filename = filename
    actual_content = content

    if actual_content is None: # Try to parse from filename if content is not explicitly given
        if '\n' in filename:
            parts = filename.split('\n', 1)
            actual_filename = parts[0]
            actual_content = parts[1]
        else: # If no newline, and content is None, it's an error or filename is the only content.
              # For a write tool, content is expected. Let's assume filename is just filename here.
            return f"Error: Content for {actual_filename} is missing. Please provide content."


    try:
        # Ensure directory exists
        directory = os.path.dirname(actual_filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(actual_filename, "w") as f:
            f.write(actual_content)
        return f"Successfully written to {actual_filename}."
    except Exception as e:
        return f"Error writing to {actual_filename}: {str(e)}"

# @tool("Reads content from a specified file. Input must be the filename.")
def read_file_tool(filename: str) -> str:
    """
    Reads content from a specified file.
    Input:
        filename (str): The name of the file to read from.
    """
    try:
        with open(filename, "r") as f:
            content = f.read()
        # For CrewAI, it's often good to return a manageable chunk of text.
        if len(content) > 5000:
            return content[:5000] + "\n... (content truncated)"
        return content if content.strip() else f"(empty file: {filename})"
    except FileNotFoundError:
        return f"Error: File '{filename}' not found."
    except Exception as e:
        return f"Error reading file {filename}: {str(e)}"
