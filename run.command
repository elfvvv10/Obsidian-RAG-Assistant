#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Local virtual environment not found at .venv/bin/activate"
  echo "Create it first, for example:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  read -r "?Press Enter to close..."
  exit 1
fi

source ".venv/bin/activate"

echo "Launching Obsidian RAG Assistant from:"
echo "  $SCRIPT_DIR"
echo

streamlit run streamlit_app.py
