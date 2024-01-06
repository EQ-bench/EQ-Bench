#!/bin/bash

# This script is for getting everything installed on a fresh linux machine (like a runpod or vast.ai environment).
# It will install oobabooga and eq-bench dependencies then launch the benchmark.

pip3 install huggingface_hub

# Get the absolute path of the current directory
ROOT_DIR=$(pwd)

# Pass the root directory to the Python script
python3 lib/quickstart.py "$ROOT_DIR"