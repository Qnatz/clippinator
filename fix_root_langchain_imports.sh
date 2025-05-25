#!/usr/bin/env bash
# File: fix_root_langchain_imports.sh
# Usage: bash fix_root_langchain_imports.sh /path/to/project

PROJECT_DIR="$1"
if [ -z "$PROJECT_DIR" ]; then
  echo "Usage: $0 /path/to/project"
  exit 1
fi

echo "Updating root LangChain imports in: $PROJECT_DIR"

find "$PROJECT_DIR" -type f -name "*.py" \
  -not -path "*/venv/*" \
  -not -path "*/.venv/*" \
  -not -path "*/__pycache__/*" | while read -r file; do
    sed -i \
      -e 's|from langchain import LLMChain|from langchain.chains import LLMChain|g' \
      -e 's|from langchain import PromptTemplate|from langchain_core.prompts import PromptTemplate|g' \
      "$file"
done

echo "Done. Please rerun your CLI to confirm warnings are gone."
