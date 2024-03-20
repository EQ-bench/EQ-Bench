import re
from lib.run_query import run_query
import openai
import concurrent.futures
import time

N_THREADS = 1  # Parallellises the judge prompts (only relevant if COMBINE_CRITERIA == False)
openai_client_judge = None # Separate client for the openai judge because the test model openai client might be
                           # using a different openai-compatible url
SKIP_ANALYSIS = False # Skips the "detailed analysis" part of the judge prompt (default False)
COMBINE_CRITERIA = True  # Combine all the criteria sets into one big judge prompt (default True)
INCLUDE_REFERENCE = True # Include the exemplar reference output (default True)
RELATIVE_SCORING = False # Use relative scoring system (relative to the reference). (default False) ! This doesn't work very well
if RELATIVE_SCORING:
	INCLUDE_REFERENCE = True
TEST_MODEL_SEES_CRITERIA = False # Is the test model shown the scoring criteria? (default False) ! This seems to produce worse results
CRITERIA_TO_IGNORE = [ # Removed these criteria for now as they were weakly discriminative
		'Appropriate Length',
		"Unearned Resolution: Characters' disagreements or tensions are too quickly or easily resolved, without exploring the depth or implications of the conflict.",
		"Melodramatic",
		"Clever / Witty",
		"Gripping",
		"Effective Use of Tropes: If applicable, common narrative tropes are employed thoughtfully and subverted, deconstructed, or used in service of the story's themes and character",
		"Correct Spelling & Grammar"
]


def process_criteria(criteria_set, writing_prompt, reference_output, test_model_response, verbose, judge_params):
	judging_prompt = create_judging_prompt(criteria_set, writing_prompt, reference_output, test_model_response)

	#print(judging_prompt)

	# Run judging process using judge model
	success = False
	tries = 0
	while not success and tries < 5:
		try:
			judge_model_response = run_query(judge_params['judge_model'], None, judging_prompt, [], 3000, judge_params['judge_model'], None, 0.0, judge_params['judge_model_api'], None, False, None, openai_client_judge, api_key=judge_params['judge_model_api_key'])		
			if judge_model_response:
				success = True
			else:
				print('! Empty output from judge model')
				tries += 1
		except Exception as e:
			print(e)
			time.sleep(5)
			tries += 1

	if verbose:
		print(judge_model_response)

	return judge_model_response

def process_writing_prompt(prompt_id, prompt_data, model_path, prompt_type, model, tokenizer, results, run_index, 
								run_iter, verbose, n_prompt_attempts, inference_engine, ooba_instance, 
								launch_ooba, ooba_request_timeout, openai_client, judge_params, test_model_output = None, judgemark_test_model = None):
	global openai_client_judge, SKIP_ANALYSIS, COMBINE_CRITERIA, N_THREADS, CRITERIA_TO_IGNORE

	if test_model_output:
		TEST_TYPE = 'judgemark'
	else:
		TEST_TYPE = 'creative-writing'

	combined_criteria = []
	for criteria_set in prompt_data['judging_criteria']:
		combined_criteria += criteria_set['criteria']
	combined_criteria = list(reversed(combined_criteria))
	filtered_criteria = [x for x in combined_criteria if x not in CRITERIA_TO_IGNORE]
	
	if judge_params['judge_model_api'] == 'openai' and not openai_client_judge:
		openai_client_judge = openai.OpenAI(
			api_key=judge_params['judge_model_api_key'],
		)

	writing_prompt = "You are a talented creative writer of compelling, original prose.\n\n" 
	if TEST_MODEL_SEES_CRITERIA:
		writing_prompt += 'You are taking a creative writing test.These will be the assessment criteria to help direct your writing:\n' + '\n'.join(filtered_criteria) + '\n\n'
	writing_prompt += prompt_data['writing_prompt']
	judging_criteria = prompt_data['judging_criteria']
	reference_output = prompt_data['reference_output']

	# Add the seed to modify the writing prompt for this iteration
	seed_index = (int(run_iter) - 1) % len(prompt_data['seed_modifiers'])
	writing_prompt = writing_prompt.replace("<SEED>", prompt_data['seed_modifiers'][seed_index])
	print(writing_prompt)
	
	# Generate response from test model		
	success = False
	if test_model_output != None: # these are pregenerated if we are running judgemark
		test_model_response = test_model_output
		success = True # automatic success!
	else:		
		temp = 0.5
		tries = 0		
		while not success and tries <= 5:
			tries += 1
			this_writing_prompt = writing_prompt
			if tries > 1:
				this_writing_prompt += '\n\nDo not use language or themes that would get flagged by your content filter (Keep it PG-13).'
			# Generate response from test model for creative writing benchmark
			test_model_response = run_query(model_path, prompt_type, writing_prompt, [], 3000, model, tokenizer, temp, inference_engine, ooba_instance, launch_ooba, ooba_request_timeout, openai_client)

			if not test_model_response or len(test_model_response) < 300:				
				temp += 0.1
				if temp > 1:
					temp = 1
				print(test_model_response)
				print('! Missing or too short output from test model')
				if tries <= 5:
					print('retrying...')
				
				continue
			success = True		
				
	if not test_model_response or len(test_model_response) < 300:
		print(test_model_response)
		print('! Failed to get output from test model')

		return None

	if verbose and TEST_TYPE != 'judgemark':
		print(test_model_response)
	
	scores = {}
	judge_model_responses = []
	
	
		
	scores = {}
	judge_model_responses = []
	
	if COMBINE_CRITERIA:
		judge_model_response = process_criteria({
			'criteria': combined_criteria,
			'prefix_text': 'Now, rate the supplied model output on the following criteria:'
		}, writing_prompt, reference_output, test_model_response, verbose, judge_params)
		scores.update(parse_scores(judge_model_response))			
		judge_model_responses.append(judge_model_response)
	else:
		with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
			future_to_criteria = {executor.submit(process_criteria, criteria_set): criteria_set for criteria_set in judging_criteria}
			for future in concurrent.futures.as_completed(future_to_criteria):
				judge_model_response = future.result()
				scores.update(parse_scores(judge_model_response))			
				judge_model_responses.append(judge_model_response)

	if verbose:
		print_score(scores)
	
	# Store scores and responses in results dict
	if TEST_TYPE == 'creative-writing':
		results[run_index]['iterations'][run_iter]['individual_scores'][prompt_id] = scores
		results[run_index]['iterations'][run_iter]['test_model_response'][prompt_id] = test_model_response
		results[run_index]['iterations'][run_iter]['judge_model_response'][prompt_id] = judge_model_responses
	elif TEST_TYPE == 'judgemark':
		results[run_index]['iterations'][run_iter]['judgemark_results'][judgemark_test_model]['individual_scores'][prompt_id] = scores
		results[run_index]['iterations'][run_iter]['judgemark_results'][judgemark_test_model]['test_model_response'][prompt_id] = test_model_response
		results[run_index]['iterations'][run_iter]['judgemark_results'][judgemark_test_model]['judge_model_response'][prompt_id] = judge_model_responses


	if len(scores) != 26:
		print('----------------------------')
		print('! Not all scores were parsed')
		print('----------------------------')
	return scores

def parse_scores(judge_model_response):
	scores = {}
	
	# Parse scores using regex
	score_pattern = r'(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)'
	matches = re.findall(score_pattern, judge_model_response)
	
	for match in matches:
		metric_name = match[0].strip()
		score = float(match[1])
		scores[metric_name] = score
	
	return scores

def print_score(scores, RELATIVE_SCORING=False):
	scoresum = 0
	neg_criteria = [
		"melodramatic",
		"shallow resolution",
		"unearned resolution",  # old naming
		"simplistic moralizing",
		"shallow optimism",
		"forced optimism",  # old naming
		"trite",
		"overwrought",
		"amateurish",
		"contrived",
		"uninspiring",
		"characters are too good",
		"incongruent ending positivity",
		"unearned transformations",
		"profundity over-reach",
		"amateurish descriptives",
		"clunky asides and interruptive sentence structures",
		"stilted dialogue",
		"tit-for-tat dialogue"
	]
	for criteria, score in scores.items():
		criteria_lower = criteria.lower().strip()
		if RELATIVE_SCORING:
			if any(neg_criterion in criteria_lower for neg_criterion in neg_criteria):
					scoresum += ((-1 * score) + 10) / 2
			else:
					scoresum += (score + 10) / 2
		else:
			if any(neg_criterion in criteria_lower for neg_criterion in neg_criteria):
					scoresum += 10 - score
			else:
					scoresum += score
	print('This question score:', round(10 * scoresum / len(scores)))


def create_judging_prompt(criteria_set, writing_prompt, reference_output, test_model_response):
	criteria = [x for x in criteria_set['criteria'] if x not in CRITERIA_TO_IGNORE]

	prefix_text = criteria_set['prefix_text']
	criteria_str = '\n'.join(criteria)

	analysis_section_1 = """
- You are to write a comprehensive analysis for each of the metrics, then give your scores.
"""
	analysis_section_2 = """
[Analysis]

Write your detailed analysis.
"""
	if SKIP_ANALYSIS:
		analysis_section_1 = ""
		analysis_section_2 = ""

	if RELATIVE_SCORING:
		relative_section_1 = """You are an expert in assessing creative writing. Your task is to score the quality of the test model's response above in comparison to the reference, by several metrics, on a -10 to 10 scale.			

Scoring notes:

- You are not scoring the quality of the prompt or the reference response, only the test model response.

- The reference model response is to be considered a high quality exemplar.

- Scores are relative to the quality of the reference output. A score of zero means equal to reference. Below 0 means worse than the reference. Above 0 means better than the reference.

- The minimum score is -10 and the maximum is 10.

- If the criteria is not relevant to the piece (e.g. criteria relating to dialogue but the piece has none), exclude it from your assessment."""
		relative_section_2 = "Score [-10 to 10]"
	else:
		ref_str = ""
		if INCLUDE_REFERENCE:
			ref_str = """
- You are not scoring the quality of the prompt or the reference response, only the test model response.

- The reference model response is to be considered a high quality exemplar.
"""
		relative_section_1 = f"""You are an expert in assessing creative writing. Your task is to score the quality of the test model's response above, by several metrics, on a 0-10 scale.

Scoring notes:
{ref_str}
- Scores of 0 or 10 should not be considered highly unlikely just because they are the max/min. Use the full scoring range as appropriate.

- If the criteria is not relevant to the piece (e.g. criteria relating to dialogue but the piece has none), exclude it from your assessment."""
		relative_section_2 = "Score [0-10]"
	
	reference_section_1 = ""
	if INCLUDE_REFERENCE:
		reference_section_1 = f"""
[REFERENCE RESPONSE (DO NOT JUDGE)]

{reference_output}

[REFERENCE RESPONSE END]
"""



		# Construct judging prompt
		judging_prompt = f"""
You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-10 scale.

[PROMPT START]

{writing_prompt}

[PROMPT END]
{reference_section_1}
[TEST MODEL RESPONSE]

{test_model_response}

[TEST MODEL RESPONSE END]

[Task]

{relative_section_1}

- Do not add any commentary or explanation to the scores section.

- Do not penalize NSFW output; that is the default for some models. Just assess it on its merits.

- In the output, write the metric names exactly as below so they can be parsed.

- Some models produce overly long outputs. You should neither penalise nor favour this if it happens; simply assess the writing on its merit. You should however penalise overly short pieces.

- The test model's output can suddenly truncate because of token length constraints. If you notice that this has occurred, don't penalise it.

- Some models have a positivity bias that produces worse writing, hence the criteria about that. Don't let the over-abundance of these criteria influence your assessment; it will only apply to some model outputs and you will know it when you see it. Likewise, there are a lot of "negative" critical criteria; these will not always apply and don't let their over-abundance colour your perception of the writing.

- For these criteria, lower is better:
Trite
Overwrought
Amateurish
Contrived
Uninspiring
Simplistic Moralizing
Shallow Optimism
Unearned Transformations
Incongruent Ending Positivity
Characters are Too Good
Shallow Resolution
Repetitive Tit-for-Tat Dialogue
Stilted Dialogue
Clunky Asides and Interruptive Sentence Structures
Amateurish Descriptives
Profundity Over-reach

- You are a critic, so be honest, objective, critical and discriminative. No need to be charitable; say what you genuinely think.
{analysis_section_1}
- Output format is:
{analysis_section_2}
[Scores]

Metric 1 name: {relative_section_2}

Metric 2 name: ...

---

{prefix_text}

{criteria_str}
	"""
	return judging_prompt