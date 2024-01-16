#!/bin/bash

# EQ-bench requirements:
pip install -q tqdm sentencepiece hf_transfer openai scipy torch peft bitsandbytes git+https://github.com/huggingface/transformers.git trl accelerate tensorboardX huggingface_hub
# These are for qwen models:
pip install einops transformers_stream_generator==0.0.4 deepspeed tiktoken git+https://github.com/Dao-AILab/flash-attention.git auto-gptq optimum
# These are for uploading results
pip install gspread oauth2client firebase_admin