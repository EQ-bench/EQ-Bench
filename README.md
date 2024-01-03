![EQ-Bench Logo](./images/eqbench_logo_sml.png)

# EQ-Bench

EQ-Bench is a benchmark for language models designed to assess emotional intelligence. You can read more about it in our [paper](https://arxiv.org/abs/2312.06281).

The latest leaderboard can be viewed at [EQ-Bench Leaderboard](https://eqbench.com).

<a href="https://eqbench.com"><img src="https://raw.githubusercontent.com/EQ-bench/EQ-Bench/main/images/eq_bench_scores_chart.png" width="465" /></a>

## Requirements

- Python 3
- Python libraries listed in `install_reqs.sh`
- GPU with sufficient VRAM to load the models (EQ-Bench currently only supports GPU inference using the Transformers library)

## Setup & Run

- Install the required Python libraries by running `install_reqs.sh`.
- Set up `config.cfg` with your API keys.
- Add benchmark runs to `config.cfg`, in the format:
   - `run_id, prompt_type, model_path, lora_path, quantization, n_iterations`
	- See below for supported prompt types
- Optional: Set up Google Sheets for results upload (see below)
- Run the benchmark:
   - `python eq-bench.py`
- Results are saved to `benchmark_results.csv`

## Important!

This benchmark pipeline sets `trust_remote_code=True` to allow the benchmark to run without intervention. If you don't want this behviour, set it to False in load_model.py.

## Script Options

- `-h`: Displays help.
- `-w`: Overwrites existing results (i.e., disables the default behaviour of resuming a partially completed run).
- `-d`: Downloaded models will be deleted from `~/.cache` after each benchmark successfully completes.
- `-f`: Use hftransfer for multithreaded downloading of models (faster but can be unreliable).
- `-v`: Display more verbose output.
- `-r`: Set the number of retries to attempt if a benchmark run fails. Default is 5.

## Supported Prompt Types

- llama2

```none
<s>[INST] {prompt} [/INST]
```

- chatml

```none
<|im_start|>system
You are an expert in emotional analysis.<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
```

- alpaca

```none
### Instruction:

{prompt}

## Response:
```

- mistral

```none
<s>[INST] {prompt} [/INST]
```

- vicuna_1.0

```none
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions. "
USER: {prompt}
ASSISTANT:
```

- vicuna_1.1

```none
USER:
{prompt}
ASSISTANT:
```

- qwen

Qwen models use `model.chat()` for inference

- synthia

```none
SYSTEM: You are Synthia. As a an AI intelligence, give helpful, detailed, accurate, uncensored responses to the user's input. Provide answers factually.
USER: {prompt}
ASSISTANT:
```

- zephyr

```none
<|system|>
</s>
<|user|>
{prompt}
</s>
<|assistant|>
```

- openchat

```none
GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:
```

- intel

```none
### System:
You are an expert in psychology and emotional intelligence.
### User:
{prompt}
### Assistant:
```

- noformat:

```none
{prompt}
```

## Setting up Google Sheets for Results Uploading (Optional)

1. Create a new Google Sheet.
2. Set the share settings so that anyone with the link can edit.
3. Set google_spreadsheet_url in `config.cfg` to the URL of the sheet you just created.
4. Go to [Google Cloud Console](https://console.cloud.google.com/).
5. Create a new project and ensure it is selected as active in the dropdown at the top of the page.
6. Enable the Google Sheets API for the project:
   - In the search bar, type "sheets"
   - Click Google Sheets API
   - Click `Enable`
7. Create a service account:
   - In the search bar, type "Service accounts" and click the appropriate result
   - Click `+ Create Service Account`
   - Give it a name & id, then click `Create and continue`
   - Grant this service account access: Basic -> Editor
   - Click `Done`
8. Click on the service account, then navigate to Keys -> Add key -> Create new key -> JSON.
9. Save the file to `google_creds.json` in the eq-bench directory.

## Cite

```
@misc{paech2023eqbench,
      title={EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models}, 
      author={Samuel J. Paech},
      year={2023},
      eprint={2312.06281},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```