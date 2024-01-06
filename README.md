![EQ-Bench Logo](./images/eqbench_logo_sml.png)

# EQ-Bench

EQ-Bench is a benchmark for language models designed to assess emotional intelligence. You can read more about it in our [paper](https://arxiv.org/abs/2312.06281).

The latest leaderboard can be viewed at [EQ-Bench Leaderboard](https://eqbench.com).

## Requirements

- Linux (and possibly mac; untested) supported
- Python 3x
- Python libraries listed in `install_reqs.sh`
- Working install of Oobabooga (optional)
- Sufficient GPU / System RAM to load the models

## Installation

### Quick start:

If you are installing EQ-Bench into a fresh linux install (like a runpod or similar), you can run `ooba_quick_install.sh`. This will install oobabooga into the current user's home directory, install all EQ-Bench dependencies, then run the benchmark pipeline.

### Install (not using quick start)

Note: Ooobabooga is optional. If you prefer to use transformers as the inference engine, or if you are only benchmarking through the OpenAI API, you can skip installing it.

- Install the required Python dependencies by running `install_reqs.sh`.
- Optional: install the [Oobabooga library](https://github.com/oobabooga/text-generation-webui/tree/main) and make sure it launches.
- Optional: Set up Google Sheets for results upload (see below)

### Configure

- Set up `config.cfg` with your API keys and runtime settings.
- Add benchmark runs to `config.cfg`, in the format:
   - `run_id, instruction_template, model_path, lora_path, quantization, n_iterations, inference_engine, ooba_params, downloader_args`

      - `run_id`: A name to identify the benchmark run
      - `instruction_template`: The filename of the instruction template defining the prompt format, minus the .yaml (e.g. Alpaca)
      - `model_path`: Huggingface model ID, local path, or OpenAI model name
      - `lora_path` (optional): Path to local lora adapter
      - `quantization`: Using bitsandbytes package (8bit, 4bit, None)
      - `n_iterations`: Number of benchmark iterations (final score will be an average)
      - `inference_engine`: Set this to transformers, openai or ooba.
      - `ooba_params` (optional): Any additional ooba params for loading this model (overrides the global setting above)
      - `downloader_filters` (optional): Specify --include or --exclude patterns (using same syntax as huggingface-cli download)

## Benchmark run examples

`# run_id, instruction_template, model_path, lora_path, quantization, n_iterations, inference_engine, ooba_params, downloader_args`

`myrun1, openai_api, gpt-4-0613, , , 1, openai, ,`

`myrun2, Llama-v2, meta-llama/Llama-2-7b-chat-hf, /path/to/local/lora/adapter, 8bit, 3, transformers, , ,`

`myrun3, Alpaca, ~/my_local_model, , None, 1, ooba, --loader transformers --n_ctx 1024 --n-gpu-layers -1, `

`myrun4, Mistral, TheBloke/Mistral-7B-Instruct-v0.2-GGUF, , None, 1, ooba, --loader llama.cpp --n-gpu-layers -1 --tensor_split 1,3,5,7, --include ["*Q3_K_M.gguf", "*.json"]`

`myrun5, Mistral, mistralai/Mistral-7B-Instruct-v0.2, , None, 1, ooba, --loader transformers --gpu-memory 12, --exclude "*.bin"`

## Running the benchmark

- Run the benchmark:
   - `python3 eq-bench.py`
- Results are saved to `benchmark_results.csv`

## Script Options

- `-h`: Displays help.
- `-w`: Overwrites existing results (i.e., disables the default behaviour of resuming a partially completed run).
- `-d`: Downloaded models will be deleted after each benchmark successfully completes. Does not affect previously downloaded models specified with a local path.
- `-f`: Use hftransfer for multithreaded downloading of models (faster but can be unreliable).
- `-v`: Display more verbose output.
- `-r`: Set the number of retries to attempt if a benchmark run fails. Default is 5.

## Prompt Formats / Instruction Templates

EQ-Bench uses the same instruction template format as the Oobabooga library. You can modify the existing ones or add your own. When you specify a prompt format in config.cfg, use the filename minus the .yaml, e.g. Alpaca.

- If using `transformers` as the inference engine, the benchmark pipeline uses templates located in `[EQ-Bench dir]/instruction-templates`.
- If using `ooba` as the inference engine, the pipeline uses templates located in `[ooba dir]/instruction-templates`

## Setting up Google Sheets for Results Uploading (Optional)

<details>
  <summary>Show instructions</summary>
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
</details>

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