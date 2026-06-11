#!/bin/zsh
# Launch the demo in the dwh-ai-py311 conda env.
# KMP_DUPLICATE_LIB_OK avoids the macOS torch/gensim duplicate-OpenMP abort.
export KMP_DUPLICATE_LIB_OK=TRUE
cd "$(dirname "$0")"
exec conda run --no-capture-output -n dwh-ai-py311 python app.py
