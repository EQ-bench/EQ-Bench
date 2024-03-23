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
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, kendalltau

# Constants
RAW_RESULTS_PATH = './raw_results.json'
BENCH_RESULTS_PATH = './benchmark_results.csv'


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


# model, arena elo, eq-bench, magi
other_benchmarks_str = """
claude-3-opus-20240229,1247,82.19,76.55
Midnight-Miqu-70B-v1.5,,75.9,40.74
claude-3-sonnet-20240229,1190,80.45,61.01
gpt-4-0125-preview,1249,83.87,76.83
claude-3-haiku-20240307,,63.35,
mistral-large-2402,1155,85.17,67.69
mistral-medium,1145,82.57,62.15
goliath-120b,,76.09,50.36
Yi-34B-Chat,1099,71.62,57.1
Qwen1.5-14B-Chat,,74.99,49.27
Mixtral-8x7B-Instruct-v0.1,1114,72.37,45.74
mistral-small,,80.36,51.9
Llama-2-13b-chat-hf,1043,49.12,28.2
Platypus2-70B-instruct,,,
openchat-3.5-1210,1071,72.52,38.81
gpt-3.5-turbo-0301,1101,70.67,46.66
Llama-2-7b-chat-hf,1027,36.32,27.5
gemma-7b-it,1029,61.72,24.85
gemma-2b-it,985,23.26,24.16
Qwen1.5-4B-Chat,974,28.75,32.66
"""

def parse_benchmarks(benchmark_str):
	from io import StringIO
	df = pd.read_csv(StringIO(benchmark_str), header=None, names=["model", "arena_elo", "eq_bench", "magi"])
	return df

def merge_benchmarks(judgemark_results, benchmark_str):
	df_judgemark = pd.DataFrame(list(judgemark_results['model_scores'].items()), columns=['model', 'judgemark'])
	df_benchmarks = parse_benchmarks(benchmark_str)
	
	# Concatenate model name after "/"
	df_judgemark['model'] = df_judgemark['model'].apply(lambda x: x.split('/')[-1])
	
	df_combined = pd.merge(df_judgemark, df_benchmarks, on='model', how='left')
	return df_combined

def calculate_correlations(data):
	correlations = {}
	for benchmark in ["arena_elo", "eq_bench", "magi"]:
		valid_data = data.dropna(subset=['judgemark', benchmark])
		if len(valid_data) > 1:  # Need at least 2 valid points to calculate correlation
			pearson_corr, _ = pearsonr(valid_data['judgemark'], valid_data[benchmark])
			kendall_corr, _ = kendalltau(valid_data['judgemark'], valid_data[benchmark])
			correlations[f'pearson_{benchmark}'] = pearson_corr
			correlations[f'kendall_{benchmark}'] = kendall_corr
		else:
			correlations[f'pearson_{benchmark}'] = None
			correlations[f'kendall_{benchmark}'] = None
	return correlations


def normalize_score(score, min_score, max_score):
	if score >= max_score:
		return 1.0
	elif score <= min_score:
		return 0.0
	else:
		return (score - min_score) / (max_score - min_score)

def calculate_metrics(data):
	metrics = {
		'mean_score': data['judgemark'].mean(),
		'range': data['judgemark'].max() - data['judgemark'].min(),
		'std_dev': data['judgemark'].std(),
		'CV': data['judgemark'].std() / data['judgemark'].mean(),
	}

	# Calculate std_dev of top 5 models
	top_5_models = data.nlargest(5, 'judgemark')
	metrics['std_dev_top_5'] = top_5_models['judgemark'].std()
	
	# Calculate correlations
	correlations = calculate_correlations(data)
	metrics.update(correlations)
	
	# Normalize metrics to 0-1 range
	normalized_metrics = {}
	for metric, value in metrics.items():
		if metric == 'mean_score':
			continue  # Skip mean, as we're leaving it out of the aggregate score
		elif metric == 'range':
			normalized_metrics[metric] = normalize_score(value, 0, 60)
		elif metric == 'std_dev':
			normalized_metrics[metric] = normalize_score(value, 0, 10)
		elif metric == 'std_dev_top_5':
			normalized_metrics[metric] = normalize_score(value, 0, 2)
		elif metric == 'CV':
			normalized_metrics[metric] = normalize_score(value, 0, 0.4)
		
	#elif metric.startswith('pearson_') or metric.startswith('kendall_'):
	#	normalized_metrics[metric] = normalize_score(value, -1, 1)
			
	kendalls_correlations = [value for key, value in metrics.items() if key.startswith('kendall_')]
	pearsons_correlations = [value for key, value in metrics.items() if key.startswith('pearson_')]
	
	avg_kendalls = sum(kendalls_correlations) / len(kendalls_correlations) if kendalls_correlations else 0
	avg_pearsons = sum(pearsons_correlations) / len(pearsons_correlations) if pearsons_correlations else 0
	
	normalized_metrics['avg_kendalls'] = normalize_score(avg_kendalls, 0, 1)
	normalized_metrics['avg_pearsons'] = normalize_score(avg_pearsons, 0, 1)

	print('# normalised:')
	for k, v in normalized_metrics.items():
		print(k,v)
	
	# Calculate aggregate score
	aggregate_score = sum(normalized_metrics.values()) / len(normalized_metrics) * 100
	metrics['aggregate_score'] = aggregate_score
	
	return metrics

def compute_judgemark_results(results, run_index, test_model_outputs, verbose):
	results[run_index]['judgemark_results'] = {}
	model_scores = {}
	for model_name, _ in test_model_outputs.items():
		# This is a placeholder for wherever you calculate the creative writing score
		creative_writing_score = calculate_creative_writing_score_judgemark(run_index, model_name, results)
		if creative_writing_score is not None:
			model_scores[model_name] = creative_writing_score
			if verbose:
					print(round(creative_writing_score, 2), model_name)
	
	mean_score = np.mean(list(model_scores.values()))
	std_dev = np.std(list(model_scores.values()), ddof=1)  # Using sample standard deviation
	
	results[run_index]['judgemark_results'] = {
		'mean_score': mean_score,
		'std_dev': std_dev,
		'model_scores': model_scores
	}

	#results[run_index]['judgemark_results']['model_scores'] = model_scores
	
	# Merge Judgemark results with other benchmarks into a DataFrame
	df_combined = merge_benchmarks(results[run_index]['judgemark_results'], other_benchmarks_str)
	
	# Calculate extended metrics
	extended_metrics = calculate_metrics(df_combined)
	

	results[run_index]['judgemark_results']['extended_metrics'] = extended_metrics

	#print(results[run_index]['judgemark_results'])
	for k,v in results[run_index]['judgemark_results']['extended_metrics'].items():
		print(k, v)
	#print(extended_metrics)
	
	# Add extended metrics to the results
	#results[run_index]['extended_metrics'] = extended_metrics



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

	save_result_to_db_fn(results[run_index], this_score, parseable if benchmark_type == 'eq-bench' else 'N/A', last_error, run_index, bench_success)


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
			print('Mean Score:', round(results[run_index]['judgemark_results']['mean_score'], 2))
			print('Std. Dev.:', round(results[run_index]['judgemark_results']['std_dev'], 2))
			print('Judgemark Score:', round(results[run_index]['judgemark_results']['extended_metrics']['aggregate_score'], 2))
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
			



