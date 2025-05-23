from __future__ import annotations

import json
import os # Added for os.path.getmtime
import subprocess
from collections import defaultdict

# Module-level cache for ctags results
ctags_cache = {}


def get_tag_kinds() -> dict[str, list[str]]:
    """
    List tags by language in decreasing order of importance
    """
    # Run "ctags --list-kinds-full"
    cmd = ["ctags", "--list-kinds-full"]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ).stdout.splitlines()[1:]
    kinds = defaultdict(list)
    for line in result:
        language, kind = line.split()[0], line.split()[2]
        kinds[language].append(kind)
    return kinds


tag_kinds_by_language = get_tag_kinds()


def get_file_summary(file_path: str, indent: str = "", length_1: int = 1000, length_2: int = 2000) -> str:
    """
    | 72| class A:
    | 80| def create(self, a: str) -> A:
    |100| class B:
    """
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        # File might not exist or be accessible, skip caching or return error
        return "" # Or handle error appropriately

    cache_key = (file_path, mtime)
    if cache_key in ctags_cache:
        print(f"[INFO] ctags cache HIT for: {file_path}")
        return ctags_cache[cache_key]
    
    print(f"[INFO] ctags cache MISS for: {file_path}. Running ctags.")
    cmd = ["ctags", "-x", "--output-format=json", "--fields=+n+l", file_path]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out_str = ""

    if result.returncode != 0:
        # Instead of raising RuntimeError, consider returning an error string or empty
        # to avoid breaking flows that expect a string summary.
        # For now, let's log to stderr if possible (though not directly from here) and return empty.
        # print(f"Error executing ctags for {file_path}: {result.stderr}", file=sys.stderr)
        return f"{indent}Error executing ctags: {result.stderr}"


    try:
        with open(file_path, "r") as f:
            file_lines = f.readlines()
    except UnicodeDecodeError:
        return "" # Cannot process if file is not readable as text

    lines_from_ctags = result.stdout.splitlines()
    if not lines_from_ctags: # Handle case where ctags produces no output (e.g. empty or non-code file)
        ctags_cache[cache_key] = ""
        return ""
        
    tags = [json.loads(line) for line in lines_from_ctags if line.strip()]
    
    if not tags: # If after parsing, there are no tags
        ctags_cache[cache_key] = ""
        return ""

    lengths_by_tag = defaultdict(int)
    for tag in tags:
        if tag['line'] > len(file_lines): # ctags line number out of bounds
             # Potentially skip this tag or log a warning
            continue
        tag['formatted'] = f"{indent}{tag['line']}|{file_lines[tag['line'] - 1].rstrip()}"
        lengths_by_tag[tag['kind']] += len(tag['formatted']) + 1
    
    if not tags[0].get('language'): # If language is not detected by ctags
        # Fallback or default behavior if language is missing
        # This might happen for plain text files or unsupported languages
        # For now, we'll attempt to proceed if kinds are available, or return empty
        kinds = []
    else:
        kinds = tag_kinds_by_language.get(tags[0]['language'], [])

    selected_tags = []
    current_length = 0
    
    # Prioritize kinds if available, otherwise try to add all tags if no kinds defined
    if kinds:
        for kind in kinds:
            kind_tags = [tag for tag in tags if tag.get('kind') == kind and 'formatted' in tag]
            for tag in kind_tags:
                if lengths_by_tag[kind] < length_1 or not selected_tags: # Original logic for length_1
                    selected_tags.append(tag)
                    current_length += len(tag['formatted']) +1 
    else: # Fallback: try to add all tags respecting overall length_2
        for tag in tags:
            if 'formatted' in tag:
                 selected_tags.append(tag)
                 current_length += len(tag['formatted']) + 1


    selected_tags = sorted(selected_tags, key=lambda tag: tag['line'])
    
    # Deduplicate lines based on line number, keeping the first encountered tag for that line
    # (ctags can sometimes produce multiple tags for the same line)
    unique_lines_dict = {}
    for tag in selected_tags:
        if tag['line'] not in unique_lines_dict:
            unique_lines_dict[tag['line']] = f"{tag['formatted']}\n"
    
    # Sort by line number before joining
    sorted_line_tuples = sorted(unique_lines_dict.items(), key=lambda item: item[0])
    
    out_list = [line_content for _, line_content in sorted_line_tuples]
    out_str = ''.join(out_list)

    if len(out_str) > length_2:
        # Ensure cut points are valid and don't split in the middle of a line marker
        # This simple trim is okay, but more robust trimming might be needed for edge cases
        out_str = out_str[:length_2 - 300] + f"\n{indent}...\n" + out_str[-300:]
    
    ctags_cache[cache_key] = out_str
    return out_str
