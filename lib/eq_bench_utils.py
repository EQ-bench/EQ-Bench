import json
from lib.scoring import calculate_score, calculate_score_fullscale, parse_answers, parse_answers_de, parse_answers_pl
from lib.run_bench_helper_functions import remove_revision_instructions
from lib.run_query import run_query
from lib.util import safe_dump

RAW_RESULTS_PATH = './raw_results.json'

def process_question(question_id, q, model_path, prompt_type, model, tokenizer, results, run_index, 
							run_iter, verbose, n_question_attempts, inference_engine, ooba_instance, 
							launch_ooba, ooba_request_timeout, openai_client, eqbench_version, language,
							REVISE):
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
 	:param language: language of the test questions ("en" default, "de" also supported)
	:return: Updated results.
	"""

	prompt = q['prompt']
	ref = q['reference_answer']
	if 'reference_answer_fullscale' in q:
		ref_fullscale = q['reference_answer_fullscale']
	else:
		ref_fullscale = None

	COMPLETION_TOKENS = 60
	if REVISE:
		COMPLETION_TOKENS = 600

	if eqbench_version == 'v2' and not REVISE:
		prompt = remove_revision_instructions(prompt, language)
		
	


	tries = 0
	success = False
	temp = 0.01 # Low temp is important for consistency of results
	prev_result = None # Stores the result of a previous partial success
	prev_result_fullscale = None
	prev_result_inference = None
	prev_result_parsed_answers = None
	while tries < n_question_attempts and not success:
		inference = run_query(model_path, prompt_type, prompt, [], COMPLETION_TOKENS, model, tokenizer, temp, inference_engine, ooba_instance, launch_ooba, ooba_request_timeout, openai_client)

		try:
			if verbose:				
				print('\n'+inference)				
				print('________________')

			# Parse and calculate scores for this question

			if language == "de":
				first_pass_answers, revised_answers = parse_answers_de(inference, REVISE)
			elif language == "pl":
				first_pass_answers, revised_answers = parse_answers_pl(inference, REVISE)
			else:
				first_pass_answers, revised_answers = parse_answers(inference, REVISE)
				
			parsed_answers = {
							'first_pass': first_pass_answers,
							'revised': revised_answers
						}

			first_pass_score = calculate_score(ref, first_pass_answers)
			if REVISE:
				revised_score = calculate_score(ref, revised_answers)
			else:
				revised_score = None
			this_result = {
				'first_pass_score': first_pass_score,
				'revised_score': revised_score
			}
			
			if ref_fullscale:
				first_pass_score_fullscale = calculate_score_fullscale(ref_fullscale, first_pass_answers)
				if REVISE:
					revised_score_fullscale = calculate_score_fullscale(ref_fullscale, revised_answers)
				else:
					revised_score_fullscale = None
				this_result_fullscale = {
					'first_pass_score': first_pass_score_fullscale,
					'revised_score': revised_score_fullscale
				}
			else:
				this_result_fullscale = {
					'first_pass_score': None,
					'revised_score': None
				}

			# Check if scores were parsed & calculated
			if first_pass_score == None or (REVISE and revised_score == None):
				if REVISE:
					if not prev_result and (first_pass_score != None or revised_score != None):
						prev_result = dict(this_result)
						prev_result_fullscale = dict(this_result_fullscale)
						prev_result_inference = inference
						prev_result_parsed_answers = dict(parsed_answers)
				raise Exception("Failed to parse scores")
			
			# Store in results dict
			results[run_index]['iterations'][run_iter]['respondent_answers'][question_id] = parsed_answers
			results[run_index]['iterations'][run_iter]['individual_scores'][question_id] = this_result
			results[run_index]['iterations'][run_iter]['individual_scores_fullscale'][question_id] = this_result_fullscale
			results[run_index]['iterations'][run_iter]['raw_inference'][question_id] = inference
			if verbose:
				if eqbench_version == 'v1':
					print('first pass:', round(first_pass_score, 1))
					if REVISE:
						print('revised:', round(revised_score, 1))
				elif eqbench_version == 'v2':
					if ref_fullscale:
						print('first pass:', round(first_pass_score_fullscale, 1))
						if REVISE:
							print('revised:', round(revised_score_fullscale, 1))

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
				results[run_index]['iterations'][run_iter]['respondent_answers'][question_id] = prev_result_parsed_answers
				results[run_index]['iterations'][run_iter]['individual_scores'][question_id] = prev_result
				results[run_index]['iterations'][run_iter]['individual_scores_fullscale'][question_id] = prev_result_fullscale
				results[run_index]['iterations'][run_iter]['raw_inference'][question_id] = prev_result_inference

	safe_dump(results, RAW_RESULTS_PATH)