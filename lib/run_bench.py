import re
import os
import time
import json
import datetime
from tqdm import tqdm
from lib.load_model import load_model
from lib.eq_bench_utils import process_question
from lib.creative_writing_utils import process_writing_prompt
from lib.scoring import calculate_eq_bench_score, calculate_creative_writing_score
from lib.db import save_eq_bench_result_to_db, save_creative_writing_result_to_db
from lib.util import upload_results_google_sheets, delete_symlinks_and_dir
from lib.run_bench_helper_functions import format_include_exclude_string, fix_results, validate_and_extract_vars, run_test_prompts, remove_revision_instructions
import lib.ooba

# Constants
RAW_RESULTS_PATH = './raw_results.json'
BENCH_RESULTS_PATH = './benchmark_results.csv'

def run_eq_bench(run_id, model_path, lora_path, prompt_type, quantization, 
                 n_iterations, resume=True, delete_cache=False, 
                 max_bench_retries=5, n_question_attempts=5, verbose=False, 
                 google_spreadsheet_url='', trust_remote_code=False, 
                 inference_engine='transformers', ooba_instance=None,
                 launch_ooba=True, cache_dir=None,
                 models_to_delete={}, models_remaining=[],
                 ooba_launch_script='', ooba_params='',
                 include_patterns=[], exclude_patterns=[],
                 ooba_params_global='', fast_download=False,
                 hf_access_token=None, ooba_request_timeout=300,
                 questions_fn=None, openai_client=None, language='en',
                 REVISE=False):
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
 	:param language: language of the test questions ("en" default, "de" also supported).
	:param REVISE: specifies whether the revision component of the prompt is included (off by default).
	"""	

	global COMPLETION_TOKENS

	with open(questions_fn, 'r') as f:
		questions = json.load(f)

	results = {}
	if resume and os.path.exists(RAW_RESULTS_PATH):
		with open(RAW_RESULTS_PATH, 'r') as f:
			results = json.load(f)
		results = fix_results(results)

	if len(questions) == 60:
		eqbench_version = "v1"
	elif len(questions) == 171:
		eqbench_version = "v2"

	if REVISE or eqbench_version == 'v1':
		# v1 must include revision
		COMPLETION_TOKENS = 1000
	else:
		COMPLETION_TOKENS = 60

	# This string is used to index this benchmark run's in the raw results dict.
	run_index = str(run_id)+'--'+eqbench_version+'--'+language+'--'+str(model_path)+'--'+str(lora_path)+'--'+str(prompt_type)+'--'+str(quantization) + '--' + inference_engine+'--'+ooba_params+'--'+format_include_exclude_string(include_patterns, exclude_patterns)
	
	# Initialise results dict
	if run_index not in results:
		results[run_index] = {}
		# Add metadata		
		run_metadata = {
			"run_id": run_id,
			"eq_bench_version": eqbench_version,
			"language": language,
			"instruction_template": prompt_type,
			"model_path": model_path,
			"lora_path": lora_path,
			"bitsandbytes_quant": quantization,
			"total_iterations": n_iterations,
			"inference_engine": inference_engine,
			"ooba_params": ooba_params,
			"include_patterns": include_patterns,
			"exclude_patterns": exclude_patterns
		}
		results[run_index]['run_metadata'] = run_metadata
	
	if 'iterations' not in results[run_index]:
		results[run_index]['iterations'] = {}
	for run_iter in range(1, n_iterations+1):
		run_iter = str(run_iter) # ensure this is always a string because json dump/load will save numeric keys as string
		if run_iter not in results[run_index]['iterations'] or not resume:
			results[run_index]['iterations'][run_iter] = {
				'respondent_answers': {},
				'individual_scores': {},
				'individual_scores_fullscale': {},
				'raw_inference': {}
			}

	# Results are only saved after all iterations are complete,
	# so we can just check the first iter to see if the run has completed.
	if resume and '1' in results[run_index]['iterations'] and 'benchmark_results' in results[run_index]['iterations']['1']:
		print('Benchmark run', run_id, 'already completed.')
		return

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
				if not model and inference_engine == 'transformers' and len(results[run_index]['iterations'][run_iter]['individual_scores']) < len(questions):
					model, tokenizer = load_model(model_path, lora_path, quantization, trust_remote_code = trust_remote_code)
				if inference_engine == 'ooba' and launch_ooba and len(results[run_index]['iterations'][run_iter]['individual_scores']) < len(questions):						
					print('Launching oobabooga...')
					ooba_instance = lib.ooba.Ooba(ooba_launch_script, model_path, cache_dir, verbose, trust_remote_code=trust_remote_code, 
													ooba_args_global=ooba_params_global, ooba_args=ooba_params, fast_download=fast_download, 
													include_patterns=include_patterns, exclude_patterns=exclude_patterns, hf_access_token=hf_access_token)
					ooba_started_ok = ooba_instance.start()
					if not ooba_started_ok:
						print('Ooba failed to launch.')
						raise Exception("Ooba failed to launch.")
					
				if model or ooba_instance:
					# This is an undocumented feature. It will run a series of test prompts for
					# each model and log the output.
					run_test_prompts(model, ooba_instance, 
							inference_engine, results, 
							model_path, prompt_type, 
							tokenizer, launch_ooba, 
							ooba_request_timeout,
							run_index, run_iter,
							verbose)

				
				# Iterate over the test questions
				for question_id, q in tqdm(questions.items()):
					if question_id in results[run_index]['iterations'][run_iter]['individual_scores']:
						if verbose:
							print('Question',question_id,'already complete')
					else:
						process_question(question_id, q, model_path, prompt_type, model, tokenizer, results, run_index, run_iter, verbose, 
                                 n_question_attempts, inference_engine, ooba_instance, launch_ooba, ooba_request_timeout, openai_client, eqbench_version,
                                 language, REVISE)
					

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
			f.write('Run ID, Benchmark Completed, Prompt Format, Model Path, Lora Path, Quantization, Benchmark Score, EQ-Bench Version, Num Questions Parseable, Num Iterations, Inference Engine, Ooba Params, Download Filters, Error\n')
	
	delete_model_files = False

	this_score = None

	lang_suffix = ''
	if language != 'en':
		lang_suffix = '_'+language
	
	parseable = None

	# Calculate final score	
	if bench_success:
		print('----Benchmark Complete----')
		print(formatted_datetime)
		print('Time taken:', round((time.time()-start_time)/60, 1), 'mins')
		print('Prompt Format:', prompt_type)
		print('Model:', model_path)
		if lora_path:
			print('Lora:', lora_path)

		# Delete model files if -d is specified and the benchmark fully completed (even if we didn't get the minimum 83% parseable answers)
		if delete_cache:
			delete_model_files = True		
		
		if eqbench_version == 'v1':
			this_score, parseable = calculate_eq_bench_score(run_index, results, RAW_RESULTS_PATH, fullscale=False)
			print('Score (v1'+lang_suffix+'):', this_score)
		else:
			this_score, parseable = calculate_eq_bench_score(run_index, results, RAW_RESULTS_PATH, fullscale=True)
			print('Score (v2'+lang_suffix+'):', this_score)
		print('Parseable:', parseable)

		if parseable / len(questions) < 0.8333:
			bench_success = False
			last_error = str(parseable) + ' questions were parseable (min is 83%)'
		else:
			this_result = [
				run_id,
				formatted_datetime,
				prompt_type,
				model_path,
				lora_path,
				quantization,
				round(this_score, 2),
				eqbench_version+lang_suffix,
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
				eqbench_version+lang_suffix,
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

	save_eq_bench_result_to_db(results[run_index], this_score, parseable, last_error, run_index, bench_success)

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


def run_creative_writing_bench(run_id, model_path, lora_path, prompt_type, quantization, 
                               n_iterations, resume=True, delete_cache=False, 
                               max_bench_retries=5, n_prompt_attempts=5, verbose=False, 
                               google_spreadsheet_url='', trust_remote_code=False, 
                               inference_engine='transformers', ooba_instance=None,
                               launch_ooba=True, cache_dir=None,
                               models_to_delete={}, models_remaining=[],
                               ooba_launch_script='', ooba_params='',
                               include_patterns=[], exclude_patterns=[],
                               ooba_params_global='', fast_download=False,
                               hf_access_token=None, ooba_request_timeout=300,
                               openai_client=None, judge_params = {}):
    
	with open('data/creative_writing_prompts.json', 'r') as f:
		writing_prompts = json.load(f)

	results = {}
	if resume and os.path.exists(RAW_RESULTS_PATH):
		with open(RAW_RESULTS_PATH, 'r') as f:
			results = json.load(f)

	run_index = str(run_id)+'--creative-writing--'+str(model_path)+'--'+str(lora_path)+'--'+str(prompt_type)+'--'+str(quantization) + '--' + inference_engine+'--'+ooba_params+'--'+format_include_exclude_string(include_patterns, exclude_patterns)
	
	if run_index not in results:
		results[run_index] = {}
		run_metadata = {
			"run_id": run_id,
			"model_path": model_path,
			"lora_path": lora_path,
			"bitsandbytes_quant": quantization,
			"total_iterations": n_iterations,
			"inference_engine": inference_engine,
			"ooba_params": ooba_params,
			"include_patterns": include_patterns,
			"exclude_patterns": exclude_patterns
		}
		results[run_index]['run_metadata'] = run_metadata
	
	if 'iterations' not in results[run_index]:
		results[run_index]['iterations'] = {}
	for run_iter in range(1, n_iterations+1):
		run_iter = str(run_iter)
		if run_iter not in results[run_index]['iterations'] or not resume:
			results[run_index]['iterations'][run_iter] = {
					'individual_scores': {},
					'test_model_response': {},
					'judge_model_response': {}
			}

	if resume and '1' in results[run_index]['iterations'] and 'benchmark_results' in results[run_index]['iterations']['1']:
		print('Creative writing benchmark run', run_id, 'already completed.')
		return

	bench_success = False
	bench_tries = 0    
	model = None
	tokenizer = None
	last_error = ''
	
	start_time = time.time()
	while not bench_success and bench_tries <= max_bench_retries:
		for run_iter in range(1, n_iterations+1):
			print('Iteration', run_iter,'of', n_iterations)
			run_iter = str(run_iter)
			try:
					if not model and inference_engine == 'transformers' and len(results[run_index]['iterations'][run_iter]['individual_scores']) < len(writing_prompts):
						model, tokenizer = load_model(model_path, lora_path, quantization, trust_remote_code = trust_remote_code)
					if inference_engine == 'ooba' and launch_ooba and len(results[run_index]['iterations'][run_iter]['individual_scores']) < len(writing_prompts):                        
						print('Launching oobabooga...')
						ooba_instance = lib.ooba.Ooba(ooba_launch_script, model_path, cache_dir, verbose, trust_remote_code=trust_remote_code, 
																ooba_args_global=ooba_params_global, ooba_args=ooba_params, fast_download=fast_download, 
																include_patterns=include_patterns, exclude_patterns=exclude_patterns, hf_access_token=hf_access_token)
						ooba_started_ok = ooba_instance.start()
						if not ooba_started_ok:
							print('Ooba failed to launch.')
							raise Exception("Ooba failed to launch.")
					
					for prompt_id, prompt_data in tqdm(writing_prompts.items()):
						if prompt_id in results[run_index]['iterations'][run_iter]['individual_scores']:
							if verbose:
									print('Prompt', prompt_id, 'already complete')
						else:
							scores = process_writing_prompt(prompt_id, prompt_data, model_path, prompt_type, model, tokenizer, results, run_index, run_iter, verbose, 
															n_prompt_attempts, inference_engine, ooba_instance, launch_ooba, ooba_request_timeout, openai_client, judge_params)
							
							if scores:
								if verbose:
									print(scores)
								with open(RAW_RESULTS_PATH, 'w') as f:
									json.dump(results, f)

					bench_success = True
			except Exception as e:
					print(e)
					last_error = ' '.join(str(e).split('\n'))
					print('Creative writing benchmark run failed.')
					bench_tries += 1
					if bench_tries <= max_bench_retries:
						print('Retrying', bench_tries,'of', max_bench_retries)

	formatted_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	
	delete_model_files = False

	this_score = None

	if bench_success:
		print('----Creative Writing Benchmark Complete----')
		print(formatted_datetime)
		print('Time taken:', round((time.time()-start_time)/60, 1), 'mins')
		print('Model:', model_path)
		if lora_path:
			print('Lora:', lora_path)

		if delete_cache:
			delete_model_files = True        
		
		this_score = calculate_creative_writing_score(run_index, results, RAW_RESULTS_PATH)
		print('Score:', this_score)

		with open(RAW_RESULTS_PATH, 'w') as f:
			json.dump(results, f)

		this_result = [
				run_id,
				formatted_datetime,
				prompt_type,
				model_path,
				lora_path,
				quantization,
				round(this_score, 2),
				'creative-writing',
				'N/A',
				n_iterations,
				inference_engine,
				ooba_params,
				format_include_exclude_string(include_patterns, exclude_patterns),
				last_error
			]
		with open(BENCH_RESULTS_PATH, 'a') as f:
			f.write(','.join(map(str, this_result)) + '\n')
	
	if not bench_success:
		print('! Creative Writing Benchmark Failed')
		print(formatted_datetime)
		print('Model:', model_path)
		if lora_path:
			print('Lora:', lora_path)
	
	save_creative_writing_result_to_db(results[run_index], this_score, last_error, run_index, bench_success)

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

def run_benchmark(run_id, model_path, lora_path, prompt_type, quantization, 
					n_iterations, resume=True, delete_cache=False, 
					max_bench_retries=5, n_question_attempts=5, n_prompt_attempts=5, verbose=False, 
					google_spreadsheet_url='', trust_remote_code=False, 
					inference_engine='transformers', ooba_instance=None,
					launch_ooba=True, cache_dir=None,
					models_to_delete={}, models_remaining=[],
					ooba_launch_script='', ooba_params='',
					include_patterns=[], exclude_patterns=[],
					ooba_params_global='', fast_download=False,
					hf_access_token=None, ooba_request_timeout=300,
					questions_fn=None, openai_client=None, language='en',
					REVISE=False, benchmark_types=[], judge_params = {}):
	
	if 'eq-bench' in benchmark_types:
		run_eq_bench(run_id, model_path, lora_path, prompt_type, quantization, 
						n_iterations, resume, delete_cache, 
						max_bench_retries, n_question_attempts, verbose, 
						google_spreadsheet_url, trust_remote_code, 
						inference_engine, ooba_instance,
						launch_ooba, cache_dir,
						models_to_delete, models_remaining,
						ooba_launch_script, ooba_params,
						include_patterns, exclude_patterns,
						ooba_params_global, fast_download,
						hf_access_token, ooba_request_timeout,
						questions_fn, openai_client, language,
						REVISE)
	
	if 'creative-writing' in benchmark_types:
		run_creative_writing_bench(run_id, model_path, lora_path, prompt_type, quantization, 
											n_iterations, resume, delete_cache, 
											max_bench_retries, n_prompt_attempts, verbose, 
											google_spreadsheet_url, trust_remote_code, 
											inference_engine, ooba_instance,
											launch_ooba, cache_dir,
											models_to_delete, models_remaining,
											ooba_launch_script, ooba_params,
											include_patterns, exclude_patterns,
											ooba_params_global, fast_download,
											hf_access_token, ooba_request_timeout,
											openai_client, judge_params)
		

