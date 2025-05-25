#!/bin/bash

# File: fix_langchain_imports.sh
# Usage: bash fix_langchain_imports.sh /path/to/project

PROJECT_DIR="$1"

if [ -z "$PROJECT_DIR" ]; then
  echo "Usage: $0 /path/to/project"
  exit 1
fi

echo "Fixing LangChain imports in: $PROJECT_DIR"
echo "----------------------------------------------------"

find "$PROJECT_DIR" -type f -name "*.py" \
  -not -path "*/venv/*" \
  -not -path "*/.venv/*" \
  -not -path "*/.git/*" \
  -not -path "*/__pycache__/*" \
  -not -path "*/.mypy_cache/*" \
  -not -path "*/.pytest_cache/*" |
while read -r file; do
  echo "Updating $file"
  sed -i \
    -e 's|from langchain.chat_models|from langchain_community.chat_models|g' \
    -e 's|from langchain.vectorstores|from langchain_community.vectorstores|g' \
    -e 's|from langchain.document_loaders|from langchain_community.document_loaders|g' \
    -e 's|from langchain.embeddings|from langchain_community.embeddings|g' \
    -e 's|from langchain.utilities|from langchain_community.utilities|g' \
    -e 's|from langchain.agents|from langchain_community.agents|g' \
    -e 's|from langchain.llms|from langchain_community.llms|g' \
    -e 's|from langchain.tools|from langchain_community.tools|g' \
    -e 's|from langchain.prompts|from langchain_core.prompts|g' \
    -e 's|from langchain.schema|from langchain_core.schema|g' \
    "$file"
done

echo "All done."
