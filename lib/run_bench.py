import re
import os
import time
import json
import datetime
from tqdm import tqdm
from lib.load_model import load_model
from lib.eq_bench_utils import process_question
from lib.creative_writing_utils import process_writing_prompt
from lib.scoring import calculate_eq_bench_score, calculate_creative_writing_score, calculate_creative_writing_score_judgemark
from lib.db import save_eq_bench_result_to_db, save_creative_writing_result_to_db, save_judgemark_result_to_db
from lib.util import upload_results_google_sheets, delete_symlinks_and_dir
from lib.run_bench_helper_functions import format_include_exclude_string, fix_results, validate_and_extract_vars, run_test_prompts, remove_revision_instructions
import lib.ooba
import numpy as np

# Constants
RAW_RESULTS_PATH = './raw_results.json'
BENCH_RESULTS_PATH = './benchmark_results.csv'
PROMPTS_TO_IGNORE = [11,12,13,14,15] + [3, 4, 16, 17, 18]


def setup_benchmark(benchmark_type, run_id, model_path, lora_path, prompt_type, quantization, inference_engine, ooba_params, include_patterns, exclude_patterns, language, judge_params, questions_fn):
	if benchmark_type == 'eq-bench':
		with open(questions_fn, 'r') as f:
			questions = json.load(f)
		process_fn = process_question
		scoring_fn = calculate_eq_bench_score
		save_result_to_db_fn = save_eq_bench_result_to_db
		eqbench_version = "v1" if len(questions) == 60 else "v2"
		run_index = f"{run_id}--{eqbench_version}--{language}--{model_path}--{lora_path}--{prompt_type}--{quantization}--{inference_engine}--{ooba_params}--{format_include_exclude_string(include_patterns, exclude_patterns)}"

	elif benchmark_type == 'creative-writing':
		with open('data/creative_writing_prompts.json', 'r') as f:
			questions = json.load(f)
		process_fn = process_writing_prompt
		scoring_fn = calculate_creative_writing_score
		save_result_to_db_fn = save_creative_writing_result_to_db
		run_index = f"{run_id}--creative-writing--{model_path}--{lora_path}--{prompt_type}--{quantization}--{inference_engine}--{ooba_params}--{format_include_exclude_string(include_patterns, exclude_patterns)}"
		eqbench_version = None

	elif benchmark_type == 'judgemark':
		with open('data/judgemark_test_set.json', 'r') as f:
			test_model_outputs = json.load(f)
		with open('data/creative_writing_prompts.json', 'r') as f:
			questions = json.load(f)
		process_fn = process_writing_prompt
		scoring_fn = calculate_creative_writing_score
		save_result_to_db_fn = save_judgemark_result_to_db
		run_index = f"{run_id}--judgemark--{judge_params['judge_model']}"
		eqbench_version = None
		
	else:
		raise ValueError(f"Invalid benchmark type: {benchmark_type}")
		
	return questions, process_fn, scoring_fn, save_result_to_db_fn, run_index, eqbench_version, test_model_outputs if benchmark_type == 'judgemark' else None


def initialize_results(run_index, benchmark_type, resume, n_iterations, run_id, model_path, lora_path, prompt_type, quantization, inference_engine, ooba_params, include_patterns, exclude_patterns, judge_params, language):
	results = {}
	if resume and os.path.exists(RAW_RESULTS_PATH):
		with open(RAW_RESULTS_PATH, 'r') as f:
			results = json.load(f)
		if benchmark_type == 'eq-bench':
			results = fix_results(results)

	if run_index not in results:
		results[run_index] = {
			'run_metadata': {
					"run_id": run_id,
					"eq_bench_version": benchmark_type,
					"total_iterations": n_iterations,
					"inference_engine": inference_engine,
					"ooba_params": ooba_params,
					"include_patterns": include_patterns,
					"exclude_patterns": exclude_patterns
			},
			'iterations': {}
		}
		if benchmark_type == 'eq-bench':
			results[run_index]['run_metadata'].update({
					"language": language,
					"instruction_template": prompt_type,
					"model_path": model_path,
					"lora_path": lora_path,
					"bitsandbytes_quant": quantization
			})
		elif benchmark_type == 'creative-writing':
			results[run_index]['run_metadata'].update({
					"model_path": model_path,
					"lora_path": lora_path,
					'judge_model': judge_params['judge_model'],
					"bitsandbytes_quant": quantization
			})
		elif benchmark_type == 'judgemark':
			results[run_index]['run_metadata'].update({
					'judge_model': judge_params['judge_model']
			})

	return results


def initialize_iterations(results, run_index, n_iterations, benchmark_type, resume):
	if 'iterations' not in results[run_index]:
		results[run_index]['iterations'] = {}
	for run_iter in range(1, n_iterations+1):
		run_iter = str(run_iter)
		if run_iter not in results[run_index]['iterations'] or not resume:
			results[run_index]['iterations'][run_iter] = {}
			if benchmark_type == 'eq-bench':
						results[run_index]['iterations'][run_iter] = {
							'respondent_answers': {},
							'individual_scores': {},
							'individual_scores_fullscale': {},
							'raw_inference': {}
						}
			elif benchmark_type == 'creative-writing':
						results[run_index]['iterations'][run_iter] = {
							'individual_scores': {},
							'test_model_response': {},
							'judge_model_response': {}
						}
			elif benchmark_type == 'judgemark':
						results[run_index]['iterations'][run_iter] = {
							'judgemark_results': {}
						}


def check_if_benchmark_complete(resume, results, run_index, benchmark_type, run_id):
	if resume and '1' in results[run_index]['iterations'] and 'benchmark_results' in results[run_index]['iterations']['1']:
		print(f"{benchmark_type} benchmark run {run_id} already completed.")
		return True
	return False


def load_model_and_launch_ooba(model_path, lora_path, quantization, inference_engine, launch_ooba, ooba_launch_script, ooba_params_global, ooba_params, fast_download, include_patterns, exclude_patterns, hf_access_token, trust_remote_code, cache_dir, verbose, results, run_index, run_iter, questions):
	model = None
	tokenizer = None
	ooba_instance = None
	if inference_engine == 'transformers' and len(results[run_index]['iterations'][run_iter]['individual_scores']) < len(questions):
		model, tokenizer = load_model(model_path, lora_path, quantization, trust_remote_code=trust_remote_code)
	if inference_engine == 'ooba' and launch_ooba and len(results[run_index]['iterations'][run_iter]['individual_scores']) < len(questions):
		print('Launching oobabooga...')
		ooba_instance = lib.ooba.Ooba(ooba_launch_script, model_path, cache_dir, verbose, trust_remote_code=trust_remote_code,
													ooba_args_global=ooba_params_global, ooba_args=ooba_params, fast_download=fast_download,
													include_patterns=include_patterns, exclude_patterns=exclude_patterns, hf_access_token=hf_access_token)
		ooba_started_ok = ooba_instance.start()
		if not ooba_started_ok:
					print('Ooba failed to launch.')
					raise Exception("Ooba failed to launch.")
	return model, tokenizer, ooba_instance

def process_questions(benchmark_type, model, ooba_instance, inference_engine, results, model_path, prompt_type, tokenizer, launch_ooba, ooba_request_timeout, run_index, run_iter, verbose, n_attempts, openai_client, questions, eqbench_version, language, REVISE, judge_params, test_model_outputs, process_fn):
	if benchmark_type == 'judgemark':
		for model_name, model_outputs in test_model_outputs.items():
			print('########################')
			print('Test model:', model_name)
			print('########################')
			model_scores = []
			if model_name not in results[run_index]['iterations'][run_iter]['judgemark_results']:
					results[run_index]['iterations'][run_iter]['judgemark_results'][model_name] = {
							'individual_scores': {},
							'test_model_response': {},
							'judge_model_response': {}
					}							
			for prompt_id, test_model_response in model_outputs.items():									
						if int(prompt_id) in PROMPTS_TO_IGNORE:
							continue
						if verbose and prompt_id in results[run_index]['iterations'][run_iter]['judgemark_results'][model_name]['individual_scores']:
							print('Prompt',prompt_id, 'already completed')
							continue
						prompt_data = questions[prompt_id]
						scores = process_fn(prompt_id, prompt_data, None, None, None, None, results, run_index,
														run_iter, verbose, 0, inference_engine, ooba_instance,
														launch_ooba, ooba_request_timeout, openai_client, judge_params,
														test_model_response, model_name)
						model_scores.append(scores)
						with open(RAW_RESULTS_PATH, 'w') as f:
							json.dump(results, f)

	else:
		for question_id, q in tqdm(questions.items()):
			if int(question_id) in PROMPTS_TO_IGNORE:
						continue
			if question_id in results[run_index]['iterations'][run_iter]['individual_scores']:
						if verbose:
							print(f"Question {question_id} already complete")
			else:
						if benchmark_type == 'eq-bench':
							process_fn(question_id, q, model_path, prompt_type, model, tokenizer, results, run_index, run_iter, verbose,
													n_attempts, inference_engine, ooba_instance, launch_ooba, ooba_request_timeout, openai_client, eqbench_version,
													language, REVISE)
						elif benchmark_type == 'creative-writing':
							scores = process_fn(question_id, q, model_path, prompt_type, model, tokenizer, results, run_index,
																	run_iter, verbose, n_attempts, inference_engine, ooba_instance, launch_ooba,
																	ooba_request_timeout, openai_client, judge_params)
							if scores:
									if verbose:
											print(scores)
									with open(RAW_RESULTS_PATH, 'w') as f:
											json.dump(results, f)


def compute_judgemark_results(results, run_index, test_model_outputs, verbose):
	results[run_index]['judgemark_results'] = {}
	model_scores = {}
	for model_name, model_outputs in test_model_outputs.items():
		
		creative_writing_score = calculate_creative_writing_score_judgemark(run_index, model_name, results)
		if creative_writing_score != None:
			model_scores[model_name] = creative_writing_score
			if verbose:
					print(round(creative_writing_score, 2), model_name)
		
	mean_score = np.mean(list(model_scores.values()))
	std_dev = np.std(list(model_scores.values()))

	results[run_index]['judgemark_results'] = {
		'mean_score': mean_score,
		'std_dev': std_dev,
		'model_scores': model_scores
	}


def save_and_upload_results(run_id, formatted_datetime, bench_success, prompt_type, model_path, lora_path, quantization, benchmark_type, lang_suffix, this_score, parseable, n_iterations, inference_engine, ooba_params, include_patterns, exclude_patterns, judge_params, results, run_index, last_error, bench_tries, max_bench_retries, google_spreadsheet_url, save_result_to_db_fn):
	if not os.path.exists(BENCH_RESULTS_PATH):
		with open(BENCH_RESULTS_PATH, 'a') as f:
			f.write('Run ID, Benchmark Completed, Prompt Format, Model Path, Lora Path, Quantization, Benchmark Score, Benchmark Version, Num Questions Parseable, Num Iterations, Inference Engine, Ooba Params, Download Filters, Error\n')

	if bench_success:
		if benchmark_type == 'eq-bench':
			this_result = [
					run_id,
					formatted_datetime,
					prompt_type,
					model_path,
					lora_path,
					quantization,
					round(this_score, 2),
					f"{benchmark_type}{lang_suffix}",
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

		elif benchmark_type == 'creative-writing':
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

		elif benchmark_type == 'judgemark':
			this_result = [
						run_id,
						formatted_datetime,
						'N/A',
						'N/A',
						'N/A',
						'N/A',
						'N/A',
						'judgemark',
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
		this_result = [
			run_id,
			formatted_datetime,
			prompt_type if benchmark_type != 'judgemark' else 'N/A',
			model_path if benchmark_type != 'judgemark' else 'N/A',
			lora_path if benchmark_type != 'judgemark' else 'N/A',
			quantization if benchmark_type != 'judgemark' else 'N/A',
			'FAILED',
			f"{benchmark_type}{lang_suffix}",
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

	save_result_to_db_fn(results[run_index], this_score, last_error if benchmark_type != 'eq-bench' else parseable, run_index, bench_success)


def cleanup(model, tokenizer, inference_engine, launch_ooba, ooba_instance, delete_model_files, model_path, include_patterns, exclude_patterns, models_to_delete, models_remaining, verbose):
	del model 
	del tokenizer
	if inference_engine == 'ooba' and launch_ooba:
		try:
			ooba_instance.stop()
		except Exception as e:
			pass

	if delete_model_files:
		this_model_key = model_path + '_' + ','.join(include_patterns) + '_' + ','.join(exclude_patterns)
		if model_path and this_model_key in models_to_delete and this_model_key not in models_remaining[1:]:
			if inference_engine == 'transformers':
						dir_to_delete = os.path.expanduser('~/.cache/huggingface/hub/models--' + model_path.replace('/', '--').replace('\\', '--'))
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


def run_generic_benchmark(run_id, model_path, lora_path, prompt_type, quantization,
										n_iterations, resume, delete_cache,
										max_bench_retries, n_attempts, verbose, 
										google_spreadsheet_url, trust_remote_code,
										inference_engine, ooba_instance,
										launch_ooba, cache_dir,
										models_to_delete, models_remaining,
										ooba_launch_script, ooba_params,
										include_patterns, exclude_patterns,
										ooba_params_global, fast_download,
										hf_access_token, ooba_request_timeout,
										questions_fn=None, openai_client=None, language='en',
										REVISE=False, benchmark_type='eq-bench', judge_params={}):

	questions, process_fn, scoring_fn, save_result_to_db_fn, run_index, eqbench_version, test_model_outputs = setup_benchmark(benchmark_type, run_id, model_path, lora_path, prompt_type, quantization, inference_engine, ooba_params, include_patterns, exclude_patterns, language, judge_params, questions_fn)

	results = initialize_results(run_index, benchmark_type, resume, n_iterations, run_id, model_path, lora_path, prompt_type, quantization, inference_engine, ooba_params, include_patterns, exclude_patterns, judge_params, language)

	initialize_iterations(results, run_index, n_iterations, benchmark_type, resume)

	if check_if_benchmark_complete(resume, results, run_index, benchmark_type, run_id):
		return

	bench_success = False  
	bench_tries = 0
	model = None
	tokenizer = None
	last_error = ''

	start_time = time.time()
	while not bench_success and bench_tries <= max_bench_retries:
		for run_iter in range(1, n_iterations+1):
			print(f"Iteration {run_iter} of {n_iterations}")
			run_iter = str(run_iter)
			try:
						if benchmark_type != 'judgemark':
							model, tokenizer, ooba_instance = load_model_and_launch_ooba(model_path, lora_path, quantization, inference_engine, launch_ooba, ooba_launch_script, ooba_params_global, ooba_params, fast_download, include_patterns, exclude_patterns, hf_access_token, trust_remote_code, cache_dir, verbose, results, run_index, run_iter, questions)

						if model or ooba_instance:
							run_test_prompts(model, ooba_instance,
															inference_engine, results,
															model_path, prompt_type,
															tokenizer, launch_ooba,
															ooba_request_timeout,
															run_index, run_iter,
															verbose)

						process_questions(benchmark_type, model, ooba_instance, inference_engine, results, model_path, prompt_type, tokenizer, launch_ooba, ooba_request_timeout, run_index, run_iter, verbose, n_attempts, openai_client, questions, eqbench_version, language, REVISE, judge_params, test_model_outputs, process_fn)
											
						if benchmark_type == 'judgemark':
							compute_judgemark_results(results, run_index, test_model_outputs, verbose)

						bench_success = True
			except Exception as e:  
						print(e)
						last_error = ' '.join(str(e).split('\n')) 
						print(f"{benchmark_type} benchmark run failed.")
						bench_tries += 1
						if bench_tries <= max_bench_retries:
							print(f"Retrying {bench_tries} of {max_bench_retries}")

	formatted_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	this_score = None
	lang_suffix = ''  
	parseable = None
	if bench_success:
		print(f"----{benchmark_type} Benchmark Complete----") 
		print(formatted_datetime)
		print('Time taken:', round((time.time()-start_time)/60, 1), 'mins')
		if benchmark_type != 'judgemark':
			print('Prompt Format:', prompt_type)
			print('Model:', model_path)
		if lora_path:
			print('Lora:', lora_path)
		delete_model_files = delete_cache

		if benchmark_type == 'eq-bench': 
			if language != 'en':
						lang_suffix = '_' + language
			if eqbench_version == 'v1':
						this_score, parseable = scoring_fn(run_index, results, RAW_RESULTS_PATH, fullscale=False)
						print(f"Score (v1{lang_suffix}):", this_score)  
			else:
						this_score, parseable = scoring_fn(run_index, results, RAW_RESULTS_PATH, fullscale=True)
						print(f"Score (v2{lang_suffix}):", this_score)
			print('Parseable:', parseable)
			if parseable / len(questions) < 0.8333:
						bench_success = False
						last_error = str(parseable) + ' questions were parseable (min is 83%)'

		elif benchmark_type == 'creative-writing':
			this_score = scoring_fn(run_index, results, RAW_RESULTS_PATH)  
			print('Creative Writing Score:', this_score)
			print('Judge:', judge_params['judge_model'])
			with open(RAW_RESULTS_PATH, 'w') as f:
						json.dump(results, f)

		elif benchmark_type == 'judgemark':
			print('Judge:', judge_params['judge_model'])
			print("Final Judgemark Benchmark Results:")
			print('Mean Score:', results[run_index]['judgemark_results']['mean_score'])
			print('Std. Dev.:', results[run_index]['judgemark_results']['std_dev'])
			with open(RAW_RESULTS_PATH, 'w') as f:
				json.dump(results, f)

	if not bench_success:
		print(f"! {benchmark_type} Benchmark Failed")
		print(formatted_datetime)
		if benchmark_type != 'judgemark':
			print('Prompt Format:', prompt_type)
			print('Model:', model_path)
			if lora_path:
						print('Lora:', lora_path)

	save_and_upload_results(run_id, formatted_datetime, bench_success, prompt_type, model_path, lora_path, quantization, benchmark_type, lang_suffix, this_score, parseable, n_iterations, inference_engine, ooba_params, include_patterns, exclude_patterns, judge_params, results, run_index, last_error, bench_tries, max_bench_retries, google_spreadsheet_url, save_result_to_db_fn)

	cleanup(model, tokenizer, inference_engine, launch_ooba, ooba_instance, delete_model_files, model_path, include_patterns, exclude_patterns, models_to_delete, models_remaining, verbose)





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
					REVISE=False, benchmark_types=[], judge_params={}):

	for benchmark_type in benchmark_types:
		if benchmark_type == 'eq-bench':
			run_generic_benchmark(run_id, model_path, lora_path, prompt_type, quantization,
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
											REVISE, benchmark_type)

		elif benchmark_type == 'creative-writing':
			run_generic_benchmark(run_id, model_path, lora_path, prompt_type, quantization,
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
											openai_client=openai_client, judge_params=judge_params,
											benchmark_type=benchmark_type)

		elif benchmark_type == 'judgemark':
			run_generic_benchmark(run_id, None, None, None, None,
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
											openai_client=openai_client, judge_params=judge_params,
											benchmark_type=benchmark_type)
			



