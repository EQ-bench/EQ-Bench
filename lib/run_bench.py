import os
import time
import json
import datetime
import shutil
from tqdm import tqdm
from lib.load_model import load_model, load_model_01ai
from lib.scoring import calculate_score, parse_answers, calculate_benchmark_score
from lib.run_query import OPENAI_CHAT_MODELS, OPENAI_COMPLETION_MODELS, MODELS_01AI, run_query
from lib.util import upload_results_google_sheets

# Constants
COMPLETION_TOKENS = 1000
RAW_RESULTS_PATH = './raw_results.json'
BENCH_RESULTS_PATH = './benchmark_results.csv'

def run_benchmark(run_id, model_path, lora_path, prompt_type, quantization, n_iterations, resume=True, delete_cache=False, max_bench_retries=5, n_question_attempts=5, verbose=False, google_spreadsheet_url=''):
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
				if model_path not in OPENAI_CHAT_MODELS and model_path not in OPENAI_COMPLETION_MODELS:
					# If this benchmark has already completed, skip loading the model
					if not model and len(results[run_index][run_iter]['individual_scores']) < 60:
						if model_path in MODELS_01AI:
							model, tokenizer = load_model_01ai(model_path, lora_path, quantization)
						else:
							model, tokenizer = load_model(model_path, lora_path, quantization)
				
				# Iterate over the 60 test questions
				for question_id, q in tqdm(questions.items()):
					if question_id in results[run_index][run_iter]['individual_scores']:
						if verbose:
							print('Question',question_id,'already complete')
					else:
						process_question(question_id, q, model_path, prompt_type, model, tokenizer, results, run_index, run_iter, verbose, n_question_attempts)
					

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
			f.write('Run ID, Benchmark Completed, Prompt Format, Model Path, Lora Path, Quantization, Benchmark Score, Num Questions Parseable, Num Iterations, Error\n')
	
	# Calculate final score
	if bench_success:
		print('----Benchmark Complete----')
		print(formatted_datetime)
		print('Time taken:', round((time.time()-start_time)/60, 1), 'mins')
		print('Prompt Format:', prompt_type)
		print('Model:', model_path)
		if lora_path:
			print('Lora:', lora_path)

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
				last_error
			]
		with open(BENCH_RESULTS_PATH, 'a') as f:
			f.write(','.join(map(str, this_result)) + '\n')
		if google_spreadsheet_url and os.path.exists('./google_creds.json'):
			upload_results_google_sheets(google_spreadsheet_url, this_result)

	# Cleanup
	del model
	del tokenizer
	if delete_cache:
		dir_to_delete = os.path.expanduser('~/.cache/huggingface/hub/models--'+model_path.replace('/', '--').replace('\\', '--'))
		if os.path.exists(dir_to_delete):
			shutil.rmtree(dir_to_delete)
			print('Cache deleted:', dir_to_delete)
		else:
			print('! Cache not found:', dir_to_delete)


def process_question(question_id, q, model_path, prompt_type, model, tokenizer, results, run_index, run_iter, verbose, n_question_attempts):
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
		inference = run_query(model_path, prompt_type, prompt, COMPLETION_TOKENS, model, tokenizer, temp)

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