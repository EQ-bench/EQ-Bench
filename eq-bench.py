import argparse
import configparser
import os
import time
from lib.util import parse_batch, is_int, preprocess_config_string, revert_placeholders_in_config, is_writing, gpu_cleanup
import lib.db
import signal
import sys
import io
import gc
import torch


ooba_instance = None

def cleanup():	
	while is_writing:
		print('Waiting for writes to complete...')
		time.sleep(0.1)
	print('All writes completed. Exiting gracefully.')


# Function to handle SIGINT
def signal_handler(sig, frame):
	global ooba_instance
	if ooba_instance:
		print('Stopping ooba...')
		ooba_instance.stop()
	# Wait a moment for any writes to finish	
	time.sleep(2)
	cleanup()
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		return False

def main():
	global ooba_instance

	# Preprocess the configuration content
	config_content = preprocess_config_string('config.cfg')
	config_file_iostream = io.StringIO(config_content)

	# Argument parser setup
	parser = argparse.ArgumentParser(description="Run benchmark pipeline based on specified configuration.")	
	parser.add_argument('-v1', action='store_true', help="Run v1 of EQ-Bench (legacy). V1 has been superseded and results are not directly comparable to v2 results.")
	parser.add_argument('-revise', action='store_true', help="Include the revision component of the test (off by default since v2.1).")
	parser.add_argument('--benchmarks', nargs='+', default=['eq-bench'],
                        help="Specify the benchmark types to run <eq-bench|creative-writing|judgemark> separated by comma.")
	parser.add_argument('-w', action='store_true',
							help="Overwrites existing results (i.e. disables the default behaviour of resuming a partially completed run).")
	parser.add_argument('-d', action='store_true',
							help="Downloaded models will be deleted after each benchmark successfully completes. Does not affect previously downloaded models specified with a local path.")
	parser.add_argument('-f', action='store_true',
							help="Use hftransfer for multithreaded downloading of models (faster but can be unreliable).")	
	parser.add_argument('-v', action='store_true',
							help="Display more verbose output.")
	parser.add_argument('-l', default='en',
							help="Set the language of the question dataset. Currently supported: en, de, pl")
	parser.add_argument('-r', type=int, default=5,
							help="Set the number of retries to attempt if a benchmark run fails. Default 5.")
	args = parser.parse_args()
	resume = not args.w

	if args.f:
		os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = '1'
	else:
		os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = '0'

	if args.revise:
		REVISE=True
	else:
		REVISE=False

	# This has to be imported AFTER hf_transfer env var is set.
	from lib.run_bench import run_benchmark
	import openai

	# Load the configuration
	# These options allow case sensitive keys, which we need to preserve the case of model paths
	config = configparser.RawConfigParser(allow_no_value=True)
	config.optionxform = str
	config.read_file(config_file_iostream)

	# Revert the placeholders to colons in the parsed configuration
	preprocessed_benchmark_runs = revert_placeholders_in_config(config['Benchmarks to run'])

	language = "en"
	if args.l:  # If language is provided via command line argument
		language = args.l.strip()
	
	if language not in ['en', 'de', 'pl']:
		raise Exception('Invalid language value specified.')
	
	questions_fn = './data/eq_bench_v2_questions_171.json'
	if args.v1:
		if language != "en":
			raise Exception('Error: Only English language is supported for EQ-Bench v1.')
		questions_fn = './data/eq_bench_v1_questions_60.json'

	if language != 'en':
		# Extracting the filename and extension
		base_filename, extension = questions_fn.rsplit('.', 1)
		# Appending language denotifier
		questions_fn = f"{base_filename}_{language}.{extension}"

	# Creative writing Judge params
	judge_params = {
		'judge_model_api': config['Creative Writing Benchmark'].get('judge_model_api', None),
		'judge_model': config['Creative Writing Benchmark'].get('judge_model', None),
		'judge_model_api_key': config['Creative Writing Benchmark'].get('judge_model_api_key', None)
	}

	# Check for OpenAI fields	
	api_key = config['OpenAI'].get('api_key', '')	
	base_url = 'https://api.openai.com/v1/'	
	alt_url = config['OpenAI'].get('openai_compatible_url', '')
	
	if alt_url:
		base_url = alt_url

	# If OpenAI credentials are provided, set them
	openai_client = None
	if api_key:
		openai_client = openai.OpenAI(
			api_key=api_key,
			base_url=base_url
		)

	# Check for huggingface access token
	hf_access_token = config['Huggingface'].get('access_token', '')
	if hf_access_token:
		# Set env var for the ooba downloader script
		os.environ["HF_TOKEN"] = hf_access_token
				
		# Login to hf hub
		from huggingface_hub import login
		login(token = hf_access_token)

	# Check for google sheets share url
	google_spreadsheet_url  = config['Results upload'].get('google_spreadsheet_url', '')

	# Check for firebase creds
	if os.path.exists('./firebase_creds.json'):
		os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath('./firebase_creds.json')
		lib.db.init_db()
	
	cache_dir = config['Huggingface'].get('cache_dir', '')
	if not cache_dir:
		cache_dir = None

	ooba_launch_script = config['Oobabooga config'].get('ooba_launch_script', '')	
	if ooba_launch_script:
		ooba_launch_script = os.path.abspath(os.path.expanduser(ooba_launch_script))
		if not os.path.exists(ooba_launch_script):
			raise Exception("Ooobabooga launch script not found in file system: " + ooba_launch_script)
	
	ooba_params_global = config['Oobabooga config'].get('ooba_params_global', '')

	launch_ooba = config['Oobabooga config'].get('automatically_launch_ooba', '')
	if not launch_ooba:
		launch_ooba = True
	else:
		if launch_ooba.lower() == 'true':
			launch_ooba = True
		elif launch_ooba.lower() == 'false':
			launch_ooba = False
		else:
			raise Exception('Invalid value for automatically_launch_ooba in config.cfg')

	trust_remote_code = config['Options'].get('trust_remote_code')
	if not trust_remote_code:
		trust_remote_code = False
	else:
		if trust_remote_code.lower() == 'true':
			trust_remote_code = True
		elif trust_remote_code.lower() == 'false':
			trust_remote_code = False
		else:
			raise Exception('Invalid trust_remote_code value in config.cfg')
		
	ooba_request_timeout = config['Oobabooga config'].get('ooba_request_timeout')
	if not ooba_request_timeout:
		ooba_request_timeout = 300
	else:
		if not is_int(ooba_request_timeout):
			raise Exception('Invalid ooba_request_timeout value in config.cfg')
		ooba_request_timeout = int(ooba_request_timeout)


	# Run benchmarks based on the config
	n_benchmarks = 0
	start_time = time.time()
	parsed_batch = parse_batch(preprocessed_benchmark_runs, ooba_launch_script, launch_ooba)

	# Make dict of models that need to be deleted
	models_to_delete = {}
	models_remaining = []
	
	for run_id, prompt_type, model_path, lora_path, quantization, n_iterations, \
	inference_engine, ooba_params, include_patterns, exclude_patterns in parsed_batch:
		if model_path and not os.path.exists(model_path):
			# We only want to delete model files if they won't be used in a later benchmark.
			# We also need to make sure we clear out the model dir if we are benchmarking the same model
			# again but with e.g. different .gguf files.
			this_model_key = model_path+'_'+','.join(include_patterns)+'_'+','.join(exclude_patterns)
			if args.d:
				models_to_delete[this_model_key] = True
			models_remaining.append(this_model_key)

	for run_id, prompt_type, model_path, lora_path, quantization, n_iterations, \
		inference_engine, ooba_params, include_patterns, exclude_patterns in parsed_batch:
		# Call the run_benchmark function
		print('--------------')
		print('Running benchmark', n_benchmarks + 1, 'of', len(parsed_batch))
		print('')
		print(model_path)
		n_benchmarks += 1
		if lora_path:
			print(lora_path)
		print('--------------')
		ooba_instance = None

		try:
			run_benchmark(run_id, model_path, lora_path, prompt_type, quantization, 
								n_iterations, resume=resume, delete_cache=args.d, 
								max_bench_retries=args.r, n_question_attempts=3, 
								verbose=args.v, google_spreadsheet_url=google_spreadsheet_url, 
								trust_remote_code=trust_remote_code, 
								inference_engine=inference_engine, ooba_instance=ooba_instance, 
								launch_ooba = launch_ooba, cache_dir=cache_dir,
								models_to_delete=models_to_delete, models_remaining=models_remaining,
								ooba_launch_script=ooba_launch_script, ooba_params=ooba_params,
								include_patterns=include_patterns, exclude_patterns=exclude_patterns,
								ooba_params_global=ooba_params_global, fast_download=args.f,
								hf_access_token=hf_access_token, ooba_request_timeout=ooba_request_timeout,
								questions_fn=questions_fn, openai_client=openai_client, language=language,
								REVISE=REVISE, benchmark_types=args.benchmarks, judge_params = judge_params)
		except KeyboardInterrupt:
			if ooba_instance:
				ooba_instance.stop()
			gpu_cleanup()
			raise
		except Exception as e:
			print(e)
			gpu_cleanup()

		if ooba_instance:
			ooba_instance.stop()
		gpu_cleanup()

		models_remaining = models_remaining[1:]

	print('---------------')
	print('Batch completed')
	print('Time taken:', round((time.time()-start_time)/60, 1), 'mins')
	print('---------------')
		

if __name__ == '__main__':
	main()
