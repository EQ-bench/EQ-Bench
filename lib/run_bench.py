import re
import os
import time
import json
import datetime
from tqdm import tqdm
from lib.load_model import load_model
from lib.scoring import calculate_score, parse_answers, calculate_benchmark_score
from lib.run_query import OPENAI_CHAT_MODELS, OPENAI_COMPLETION_MODELS, run_query
from lib.util import upload_results_google_sheets, delete_symlinks_and_dir
import lib.ooba

# Constants
COMPLETION_TOKENS = 1000
RAW_RESULTS_PATH = './raw_results.json'
BENCH_RESULTS_PATH = './benchmark_results.csv'

def run_benchmark(run_id, model_path, lora_path, prompt_type, quantization, 
                  n_iterations, resume=True, delete_cache=False, 
                  max_bench_retries=5, n_question_attempts=5, verbose=False, 
                  google_spreadsheet_url='', trust_remote_code=False, 
                  inference_engine='transformers', ooba_instance=None,
						launch_ooba=True, cache_dir=None,
						models_to_delete={}, models_remaining=[],
						ooba_launch_script='', ooba_params='',
						include_patterns=[], exclude_patterns=[],
						ooba_params_global='', fast_download=False,
						hf_access_token=None, ooba_request_timeout=300):
	"""
	Run a benchmark with the specified parameters.
	:param run_id: The ID string of the benchmark to be run.
	:param model_path: Path to the model.
	:param lora_path: Path to the LoRA adapter.
	:param prompt_type: The type of prompt to use.
	:param quantization: Model quantization string [4bit, 8bit, None].
	:param n_iterations: Number of iterations to run, with average taken of results.
	:param resume: Resume from last saved state if True.
	:param delete_cache: Delete downloaded model from cache after running if True.
	:param max_bench_retries: Maximum number of retries if benchmark run fails.
	:param n_question_attempts: Number of attempts per question.
	:param verbose: Verbose output if True.
	:param google_spreadsheet_url: URL for Google spreadsheet for results uploading.
	"""
	
	# This string is used to index this benchmark run's in the raw results dict.
	run_index = str(run_id)+'--'+str(model_path)+'--'+str(lora_path)+'--'+str(prompt_type)+'--'+str(quantization)

	results = {}
	if resume and os.path.exists(RAW_RESULTS_PATH):
		with open(RAW_RESULTS_PATH, 'r') as f:
			results = json.load(f)
		results = fix_results(results)
	
	# Initialise results dict
	if run_index not in results:
			results[run_index] = {}
	for run_iter in range(1, n_iterations+1):
		run_iter = str(run_iter) # ensure this is always a string because json dump/load will save numeric keys as string
		if run_iter not in results[run_index] or not resume:
			results[run_index][run_iter] = {
				'individual_scores': {},
				'raw_inference': {}
			}

	with open('data/eq_bench_questions_final.json', 'r') as f:
		questions = json.load(f)

	# Initialise vars
	bench_success = False
	bench_tries = 0	
	model = None
	tokenizer = None
	last_error = ''
	
	# Run all benchmark iterations for this model
	start_time = time.time()
	while not bench_success and bench_tries <= max_bench_retries:
		for run_iter in range(1, n_iterations+1):
			print('Iteration', run_iter,'of', n_iterations)
			run_iter = str(run_iter) # Ensure this is always a string because json dump/load will save numeric keys as string
			try:
				# Only load the model if this benchmark hasn't already completed
				if not model and inference_engine == 'transformers' and len(results[run_index][run_iter]['individual_scores']) < 60:
					model, tokenizer = load_model(model_path, lora_path, quantization, trust_remote_code = trust_remote_code)
				if inference_engine == 'ooba' and launch_ooba and len(results[run_index][run_iter]['individual_scores']) < 60:						
					print('Launching oobabooga...')
					ooba_instance = lib.ooba.Ooba(ooba_launch_script, model_path, cache_dir, verbose, trust_remote_code=trust_remote_code, 
													ooba_args_global=ooba_params_global, ooba_args=ooba_params, fast_download=fast_download, 
													include_patterns=include_patterns, exclude_patterns=exclude_patterns, hf_access_token=hf_access_token)
					ooba_started_ok = ooba_instance.start()
					if not ooba_started_ok:
						print('Ooba failed to launch.')
						raise Exception("Ooba failed to launch.")
					
				if model or ooba_instance:
					run_test_prompts(model, ooba_instance, 
							inference_engine, results, 
							 model_path, prompt_type, 
							 tokenizer, launch_ooba, 
							 ooba_request_timeout,
							 run_index, run_iter,
							 verbose)

				
				# Iterate over the 60 test questions
				for question_id, q in tqdm(questions.items()):
					if question_id in results[run_index][run_iter]['individual_scores']:
						if verbose:
							print('Question',question_id,'already complete')
					else:
						process_question(question_id, q, model_path, prompt_type, model, tokenizer, results, run_index, run_iter, verbose, 
							  n_question_attempts, inference_engine, ooba_instance, launch_ooba, ooba_request_timeout)
					

				bench_success = True
			except Exception as e:
				print(e)
				last_error = ' '.join(str(e).split('\n'))
				print('Benchmark run failed.')
				bench_tries += 1
				if bench_tries <= max_bench_retries:
					print('Retrying',bench_tries,'of',max_bench_retries)

	formatted_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	if not os.path.exists(BENCH_RESULTS_PATH):
		with open(BENCH_RESULTS_PATH, 'a') as f:
			f.write('Run ID, Benchmark Completed, Prompt Format, Model Path, Lora Path, Quantization, Benchmark Score, Num Questions Parseable, Num Iterations, Inference Engine, Ooba Params, Download Filters, Error\n')
	
	delete_model_files = False

	# Calculate final score	
	if bench_success:
		print('----Benchmark Complete----')
		print(formatted_datetime)
		print('Time taken:', round((time.time()-start_time)/60, 1), 'mins')
		print('Prompt Format:', prompt_type)
		print('Model:', model_path)
		if lora_path:
			print('Lora:', lora_path)

		# Delete model files if -d is specified and the benchmark fully completed (even if we didn't get 50 parseable answers)
		if delete_cache:
			delete_model_files = True

		benchmark_score, parseable = calculate_benchmark_score(run_index, results, RAW_RESULTS_PATH)

		if parseable < 50:
			bench_success = False
			last_error = str(parseable) + ' questions were parseable (min is 50)'
		else:
			this_result = [
				run_id,
				formatted_datetime,
				prompt_type,
				model_path,
				lora_path,
				quantization,
				round(benchmark_score, 2),
				parseable,
				n_iterations,
				inference_engine,
				ooba_params,
				format_include_exclude_string(include_patterns, exclude_patterns),
				''
			]
			with open(BENCH_RESULTS_PATH, 'a') as f:
				f.write(','.join(map(str, this_result)) + '\n')
			if google_spreadsheet_url and os.path.exists('./google_creds.json'):
				upload_results_google_sheets(google_spreadsheet_url, this_result)

	if not bench_success:
		print('! Benchmark Failed')
		print(formatted_datetime)
		print('Prompt Format:', prompt_type)
		print('Model:', model_path)
		if lora_path:
			print('Lora:', lora_path)
		this_result = [
				run_id,
				formatted_datetime,
				prompt_type,
				model_path,
				lora_path,
				quantization,
				'FAILED',
				'FAILED',
				n_iterations,
				inference_engine,
				ooba_params,
				format_include_exclude_string(include_patterns, exclude_patterns),
				last_error
			]
		with open(BENCH_RESULTS_PATH, 'a') as f:
			f.write(','.join(map(str, this_result)) + '\n')
		if google_spreadsheet_url and os.path.exists('./google_creds.json'):
			upload_results_google_sheets(google_spreadsheet_url, this_result)

	# Cleanup
	del model
	del tokenizer
	if inference_engine == 'ooba' and launch_ooba:
		try:
			ooba_instance.stop()
		except Exception as e:
			pass

	if delete_model_files:
		this_model_key = model_path+'_'+','.join(include_patterns)+'_'+','.join(exclude_patterns)
		if model_path and this_model_key in models_to_delete and this_model_key not in models_remaining[1:]:
			if inference_engine == 'transformers':
				dir_to_delete = os.path.expanduser('~/.cache/huggingface/hub/models--'+model_path.replace('/', '--').replace('\\', '--'))
				if os.path.exists(dir_to_delete):
					delete_symlinks_and_dir(dir_to_delete, verbose)
				else:
					print('! Cache not found:', dir_to_delete)
			elif inference_engine == 'ooba':
				if ooba_instance and ooba_instance.model_downloaded_fullpath:
					dir_to_delete = ooba_instance.model_downloaded_fullpath
					if os.path.exists(dir_to_delete):
						delete_symlinks_and_dir(dir_to_delete, verbose)
					else:
						print('! Cache not found:', dir_to_delete)

def format_include_exclude_string(include_patterns, exclude_patterns):
	outstr = ''
	if include_patterns:
		outstr += '--include ["'
		outstr += '", "'.join(include_patterns)
		outstr += '"] '		
	if exclude_patterns:
		outstr += '--exclude ["'
		outstr += '", "'.join(exclude_patterns)
		outstr += '"]'
	return outstr.strip()

def process_question(question_id, q, model_path, prompt_type, model, tokenizer, results, run_index, 
							run_iter, verbose, n_question_attempts, inference_engine, ooba_instance, 
							launch_ooba, ooba_request_timeout):
	"""
	Process a single question and update the results.
	:param question_id: ID of the question.
	:param q: Question data.
	:param model_path: Path to the model.
	:param prompt_type: Type of the prompt.
	:param model: Loaded model.
	:param tokenizer: Loaded tokenizer.
	:param results: Results dictionary to update.
	:param run_index: Index of the current run.
	:param run_iter: Current iteration.
	:param verbose: Verbose output flag.
	:param n_question_attempts: Number of attempts per question.
	:return: Updated results.
	"""

	prompt = q['prompt']
	ref = q['reference_answer']

	tries = 0
	success = False
	temp = 0.01 # Low temp is important for consistency of results
	prev_result = None # Stores the result of a previous partial success
	prev_result_inference = None
	while tries < n_question_attempts and not success:
		inference = run_query(model_path, prompt_type, prompt, [], COMPLETION_TOKENS, model, tokenizer, temp, inference_engine, ooba_instance, launch_ooba, ooba_request_timeout)

		try:
			if verbose:
				print(inference)
				print('________________')

			# Parse and calculate scores for this question
			first_pass_answers, revised_answers = parse_answers(inference)
			first_pass_score = calculate_score(ref, first_pass_answers)
			revised_score = calculate_score(ref, revised_answers)
			this_result = {
				'first_pass_score': first_pass_score,
				'revised_score': revised_score
			}

			# Check if scores were parsed & calculated
			if first_pass_score == None or revised_score == None:
				if not prev_result and (first_pass_score != None or revised_score != None):
					prev_result = dict(this_result)
					prev_result_inference = inference
				raise Exception("Failed to parse scores")
			
			# Store in results dict
			results[run_index][run_iter]['individual_scores'][question_id] = this_result
			results[run_index][run_iter]['raw_inference'][question_id] = inference
			if verbose:
				print('first pass:', round(first_pass_score, 1))
				print('revised:', round(revised_score, 1))
			success = True
		except KeyboardInterrupt:
			raise  # Re-raising the KeyboardInterrupt exception
		except Exception as e:
			print(e)				
			tries += 1

			# Increase temp before trying again for a parseable result
			temp += 0.15

			if tries < n_question_attempts:
				print('Retrying...')
			elif prev_result:
				# We are out of retries and we have a partial result, so store it in the results dict
				results[run_index][run_iter]['individual_scores'][question_id] = prev_result
				results[run_index][run_iter]['raw_inference'][question_id] = prev_result_inference

	with open(RAW_RESULTS_PATH, 'w') as f:
		json.dump(results, f)

def fix_results(results):
	"""
	Fix the results dict, as scores are sometimes erroneously parsed as lists when loading from json.
	:param results: The results dictionary to fix.
	:return: The fixed results.
	"""	
	for i, run in results.items():
		if 'individual_scores' in run:
			for question_id, scores in run['individual_scores'].items():
				if 'first_pass_score' in scores:
					if isinstance(scores['first_pass_score'], list) and len(scores['first_pass_score']) == 1:
						scores['first_pass_score'] = scores['first_pass_score'][0]
					if isinstance(scores['revised_score'], list) and len(scores['revised_score']) == 1:
						scores['revised_score'] = scores['revised_score'][0]
	return results

def validate_and_extract_vars(input_str):
	# Define the regex patterns for NAME, TEMP, and COMPLETION_TOKENS
	name_pattern = r"NAME=([a-zA-Z0-9\s:]+)\n"
	temp_pattern = r"TEMP=([0-9]*\.?[0-9]+)\n"
	tokens_pattern = r"COMPLETION_TOKENS=(\d+)\n"

	# Search for matches in the input string
	name_match = re.search(name_pattern, input_str)
	temp_match = re.search(temp_pattern, input_str)
	tokens_match = re.search(tokens_pattern, input_str)

	# Check if all matches are found
	if name_match and temp_match and tokens_match:
		# Extract values
		name = name_match.group(1)
		temp = float(temp_match.group(1))
		tokens = int(tokens_match.group(1))
		return name, temp, tokens
	else:
		raise ValueError("Required variables not found or in incorrect format")

# Example usage
input_string = "NAME=Prompt sequence 2: Coding questions\nTEMP=0.5\nCOMPLETION_TOKENS=1000"
try:
	name, temp, tokens = validate_and_extract_vars(input_string)
	print(f"NAME: {name}, TEMP: {temp}, COMPLETION_TOKENS: {tokens}")
except ValueError as e:
	print(e)


def run_test_prompts(model, ooba_instance, 
							inference_engine, results, 
							 model_path, prompt_type, 
							 tokenizer, launch_ooba, 
							 ooba_request_timeout,
							 run_index, run_iter,
							 verbose):
	if inference_engine == 'transformers':
		print('! Custom test prompts only support ooba or openai as the inference engine.')
		return
	if 'test_prompts_results' in results[run_index][run_iter]:
		return
	if not os.path.exists('./test_prompts.txt'):
		return
	
	results[run_index][run_iter]['test_prompts_results'] = {}
	try:
		with open('./test_prompts.txt', 'r') as f:
			prompts_str = f.read()

		print('Running test prompts...')
		prompt_sequences = prompts_str.split('###')
		for ps in prompt_sequences:
			if not ps.split():
				continue
			prompts = ps.split('---')
			sequence_name, temp, completion_tokens = validate_and_extract_vars(prompts[0])
			print('Prompt sequence:', sequence_name)
			results[run_index][run_iter]['test_prompts_results'][sequence_name] = []
			history = []
			for p in prompts[1:]:
				if not p.strip():
					continue
				tries=0
				success=0
				while tries < 5 and not success:
					try:
						if verbose:
							print('#####')
							print(p.strip())
							print('#####')
						inference = run_query(model_path, prompt_type, p.strip(), history, completion_tokens, model, tokenizer, temp, inference_engine, ooba_instance, launch_ooba, ooba_request_timeout)

						if inference:
							success=True
							if verbose:
								print(inference)
					except Exception as e:
						print(e)
						tries += 1

				if success:
					results[run_index][run_iter]['test_prompts_results'][sequence_name].append(p.strip())
					history.append({"role": "user", "content": p.strip()})
					results[run_index][run_iter]['test_prompts_results'][sequence_name].append(inference.strip())
					history.append({"role": "assistant", "content": inference.strip()})

			with open('test_prompts_results.txt', 'a') as f:
				out_str = '\n\n### ' + run_index + '\n\n'
				out_str += 'NAME=' + sequence_name + '\n'
				out_str += 'TEMP=' + str(temp) + '\n'
				out_str += 'COMPLETION_TOKENS=' + str(completion_tokens) + '\n\n'
				out_str += '---\n'
				out_str += '\n\n---\n\n'.join(results[run_index][run_iter]['test_prompts_results'][sequence_name])
				f.write(out_str)

	except Exception as e:
		print(e)
		print('! Failed to run test prompts.')
		