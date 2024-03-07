@echo off

REM Note that we need Python 3.10 at the moment, because there is no Windows triton wheel yet for 3.11 
REM One easy way of installing Python 3.10 is per Microsoft Store

REM for proper CUDA support:
pip3.10 install torch torchvision torchaudio --trusted-host download.pytorch.org --index-url https://download.pytorch.org/whl/cu121
REM pip3.10 install torch torchvision torchaudio --trusted-host download.pytorch.org --index-url https://download.pytorch.org/whl/cu118

REM alternatively, but at the moment not working: CUDA extension not installed, maybe without --pre?
REM pip3.10 install --force-reinstall --pre torch torchtext torchvision torchaudio torchrec --extra-index-url https://download.pytorch.org/whl/nightly/cu121

REM but FlashAttentionV2 not supported yet on Windows!

REM (other) EQ-bench requirements:
pip3.10 install pexpect optimum tiktoken tqdm sentencepiece hf_transfer openai scipy peft bitsandbytes git+https://github.com/huggingface/transformers.git trl accelerate tensorboardX huggingface_hub
pip3.10 install "https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl"

REM This is a temporary workaround for AttributeError: 'LlamaRotaryEmbedding' object has no attribute 'cos_cached':
REM pip3.10 install transformers==4.37.2

REM against recent error im mpmath:
REM pip3.10 install mpmath==1.3.0

REM These are for qwen models:
pip3.10 install einops transformers_stream_generator==0.0.4 deepspeed tiktoken git+https://github.com/Dao-AILab/flash-attention.git auto-gptq optimum

REM These are for uploading results:
pip3.10 install gspread oauth2client firebase_admin

REM These are for GPTQ support:
pip3.10 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

REM These are for using poe.com
pip3.10 install fastapi_poe
