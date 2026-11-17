"""Run a script with another interpreter (2026-11-17). run_chain.py starts every step with its own sys.executable
(R/env/rffi); the GPT-2 baseline needs R/env/qwen, which has transformers.
usage: python with_py.py PYTHON SCRIPT [ARGS...]
"""
import os
import sys

os.execv(sys.argv[1], sys.argv[1:])
