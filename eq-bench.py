import argparse
import configparser
import os
import time
from lib.util import parse_batch
import lib.ooba
import signal
import sys

ooba_instance = None

# Function to handle SIGINT
def signal_handler(sig, frame):
	global ooba_instance
	if ooba_instance:
		print('Stopping ooba...')
		ooba_instance.stop()
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

	# Argument parser setup
	parser = argparse.ArgumentParser(description="Run benchmark pipeline based on specified configuration.")
	parser.add_argument('-w', action='store_true',
							help="Overwrites existing results (i.e. disables the default behaviour of resuming a partially completed run).")
	parser.add_argument('-d', action='store_true',
							help="Downloaded models will be deleted from ~/.cache after each benchmark successfully completes.")
	parser.add_argument('-f', action='store_true',
							help="Use hftransfer for multithreaded downloading of models (faster but can be unreliable).")	
	parser.add_argument('-v', action='store_true',
							help="Display more verbose output.")
	parser.add_argument('-r', type=int, default=5,
							help="Set the number of retries to attempt if a benchmark run fails. Default 5.")
	args = parser.parse_args()
	resume = not args.w

	if args.f:
		os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = '1'
	else:
		os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = '0'	

	# This has to be imported AFTER hf_transfer env var is set.
	from lib.run_bench import run_benchmark
	from lib.run_query import openai

	# Load the configuration
	# These options allow case sensitive keys, which we need to preserve the case of model paths
	config = configparser.RawConfigParser(allow_no_value=True)
	config.optionxform = str
	config.read('config.cfg')

	# Check for OpenAI fields
	organization_id = config['OpenAI'].get('organization_id', '')
	api_key = config['OpenAI'].get('api_key', '')

	# If OpenAI credentials are provided, set them
	if api_key:
		openai.api_key = api_key
	if organization_id:
		openai.organization = organization_id

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

	# Note: if using transformers as inference engine, we will always use the default ~/.cache/huggingface/hub
	models_dir = config['Oobabooga config'].get('models_dir', '')
	if not models_dir:
		models_dir = None

	ooba_launch_script = config['Oobabooga config'].get('ooba_launch_script', '')	
	if ooba_launch_script:
		ooba_launch_script = os.path.abspath(os.path.expanduser(ooba_launch_script))
		if not os.path.exists(ooba_launch_script):
			raise Exception("Ooobabooga launch script not found in file system: " + ooba_launch_script)
	
	ooba_params_global = config['Oobabooga config'].get('ooba_params_global', '')	

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

	# Run benchmarks based on the config
	n_benchmarks = 0
	start_time = time.time()
	parsed_batch = parse_batch(config['Benchmarks to run'], ooba_launch_script)
	for run_id, prompt_type, model_path, lora_path, quantization, n_iterations, inference_engine, ooba_params in parsed_batch:
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
		if inference_engine == 'oobabooga':
			print('Launching oobabooga...')
			ooba_instance = lib.ooba.Ooba(ooba_launch_script, model_path, models_dir, trust_remote_code=trust_remote_code, ooba_args_global=ooba_params_global, ooba_args=ooba_params, fast_download=args.f)
			ooba_started_ok = ooba_instance.start()
			if not ooba_started_ok:
				print('Ooba failed to launch.')
				continue

		try:
			run_benchmark(run_id, model_path, lora_path, prompt_type, quantization, 
								n_iterations, resume=resume, delete_cache=args.d, 
								max_bench_retries=args.r, n_question_attempts=3, 
								verbose=args.v, google_spreadsheet_url=google_spreadsheet_url, 
								trust_remote_code=trust_remote_code, 
								inference_engine=inference_engine, ooba_instance=ooba_instance)
			
			if inference_engine == 'oobabooga':
				try:
					ooba_instance.stop()
				except Exception as e:
					pass
		except KeyboardInterrupt:
			if inference_engine == 'oobabooga':
				try:
					ooba_instance.stop()
				except Exception as e:
					pass
			raise

	print('---------------')
	print('Batch completed')
	print('Time taken:', round((time.time()-start_time)/60, 1), 'mins')
	print('---------------')
		

if __name__ == '__main__':
	main()