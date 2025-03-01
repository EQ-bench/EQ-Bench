import re
import math
from lib.util import safe_dump

# Parse the emotion intensity ratings from the raw inference text
def parse_answers(text, REVISE):
	first_pass_answers = {}
	revised_answers = {}

	# Strip out markdown
	text = text.replace('*', '').replace('#', '')

	# Extracting first pass answers
	if REVISE:
		first_pass_match = re.search(r'First pass scores:(.*?)Revised scores:', text, re.DOTALL)
		if first_pass_match:
			first_pass_text = first_pass_match.group(1)
			first_pass_answers = dict(re.findall(r'(\w+):\s+(\d+)', first_pass_text))

		# Extracting revised answers
		revised_match = re.search(r'Revised scores:(.*?)$', text, re.DOTALL)
		if revised_match:
			revised_text = revised_match.group(1)
			revised_answers = dict(re.findall(r'(\w+):\s+(\d+)', revised_text))
	else:
		first_pass_answers = dict(re.findall(r'(\w+):\s+(\d+)', text))
		revised_answers = {}

	return first_pass_answers, revised_answers

# we parse answers in German language ("de")
def parse_answers_de(text, REVISE):
	#print("Using german parsing.")
	first_pass_answers = {}
	revised_answers = {}

	# Strip out markdown
	text = text.replace('*', '').replace('#', '')

	first_pass_heading_pattern = r'(Erste.*?):\s*(.*?)(?=Überarbeitete|$)'
	revised_heading_pattern = r'(Überarbeitete.*?):\s*(.*)'

	if REVISE:
		first_pass_match = re.search(first_pass_heading_pattern, text, re.IGNORECASE | re.DOTALL)
		if first_pass_match:
			first_pass_text = first_pass_match.group(2)
			pairs = re.findall(r'([a-zA-ZäöüßÄÖÜ\s]+):\s*\**(\d+(?:,\d+)?)\**', first_pass_text)
			first_pass_answers = {label.strip(): score.replace('*', '') for label, score in pairs}

		revised_match = re.search(revised_heading_pattern, text, re.IGNORECASE | re.DOTALL)
		if revised_match:
			revised_text = revised_match.group(2)
			pairs = re.findall(r'([a-zA-ZäöüßÄÖÜ\s]+):\s*\**(\d+(?:,\d+)?)\**', revised_text)
			revised_answers = {label.strip(): score.replace('*', '') for label, score in pairs}
	else:
		pairs = re.findall(r'([a-zA-ZäöüßÄÖÜ\s]+):\s*\**(\d+(?:,\d+)?)\**', text)
		first_pass_answers = {label.strip(): score.replace('*', '') for label, score in pairs}
		revised_answers = {}

	return first_pass_answers, revised_answers


# we parse answers in Polish language ("pl")
def parse_answers_pl(text, REVISE):
	first_pass_answers = {}
	revised_answers = {}

	# Strip out markdown
	text = text.replace('*', '').replace('#', '')

	parsing_regexp = re.compile(r'([\w ]+): *<?(\d+)>?')
	# Extracting first pass answers
	if REVISE:
		first_pass_match = re.search(r'Pierwsze oceny:(.*?)Zmienione oceny:', text, re.DOTALL)
		if first_pass_match:
			first_pass_text = first_pass_match.group(1)
			first_pass_answers = dict(parsing_regexp.findall(first_pass_text))

		# Extracting revised answers
		revised_match = re.search(r'Zmienione oceny:(.*?)$', text, re.DOTALL)
		if revised_match:
			revised_text = revised_match.group(1)
			revised_answers = dict(parsing_regexp.findall(revised_text))
	else:
		first_pass_answers = dict(parsing_regexp.findall(text))
		revised_answers = {}

	return first_pass_answers, revised_answers

# Calculate the score for an individual question using v2 scoring system
def calculate_score_fullscale(reference, user):
	# First check that the emotions specified in the answer match those in the reference
	if len(user.items()) != 4:
		#print('! Error: 4 emotions were not returned')
		#print(user)
		return None
	emotions_dict = {}
	for emotion, user_emotion_score in user.items():
		for i in range(1, 5):
			if emotion.lower() == reference[f'emotion{i}'].lower():
				emotions_dict[emotion.lower()] = True
	if len(emotions_dict) != 4:
		print('! Error: emotions did not match reference')
		print(user)
		return None
	
	difference_tally = 0  # Tally of differerence from reference answers for this question
	
	# Iterate over each emotion in the user's answers.
	for emotion, user_emotion_score in user.items():
		# If this emotion is in the reference, calculate the difference between the user's score and the reference score.
		for i in range(1, 5):
			if emotion.lower() == reference[f'emotion{i}'].lower():
				d = abs(float(user_emotion_score) - float(reference[f'emotion{i}_score']))
				# this will be a value between 0 and 10
				if d == 0:
					scaled_difference = 0
				elif d <= 5:
					# S-shaped scaling function
					# https://www.desmos.com/calculator
					# 6.5\cdot\ \frac{1}{\left(1\ +\ e^{\left(-1.2\cdot\left(x-4\right)\right)}\right)}						
					scaled_difference = 6.5 * (1 / (1 + math.e ** (-1.2 * (d-4))))

				else:
					scaled_difference = d
				difference_tally += scaled_difference
					
	# Inverting the difference tally so that the closer the answer is to reference, the higher the score.
	# The adjustment constant is chosen such that answering randomly produces a score of zero.
	adjust_const =  0.7477
	final_score = 10 - (difference_tally * adjust_const)
	
	return final_score

# Calculate the score for an individual question (Legacy v1 scoring)
def calculate_score(reference, user):
	# First check that the emotions specified in the answer match those in the reference
	if len(user.items()) != 4:
		print('! Error: 4 emotions were not returned')
		print(user)
		return None
	emotions_dict = {}
	for emotion, user_emotion_score in user.items():
		for i in range(1, 5):
			if emotion.lower() == reference[f'emotion{i}'].lower():
				emotions_dict[emotion] = True
	if len(emotions_dict) != 4:
		print('! Error: emotions did not match reference')
		print(user)
		return None
	
	# Normalize the user's scores to sum to 10.
	total_user_score = sum(float(score) for score in user.values())
	if total_user_score <= 0:
		print('Error: total of scores must be > 0')
		print(user)
		return None
	user = {emotion: float(score) / total_user_score * 10 for emotion, score in user.items()}
	
	difference_tally = 0  # Tally of differerence from reference answers for this question
	
	# Iterate over each emotion in the user's answers.
	for emotion, user_emotion_score in user.items():
		# If this emotion is in the reference, calculate the difference between the user's score and the reference score.
		for i in range(1, 5):
			if emotion == reference[f'emotion{i}']:
					difference_tally += abs(user_emotion_score - reference[f'emotion{i}_score'])
					
	# Inverting the difference tally so that the closer the answer is to reference, the higher the score.
	# We subtract from 10 because it works out that this constant produces a score of 0 when answering
	# randomly, which is a useful floor for the benchmark.
	final_score = 10 - difference_tally
	
	return final_score

# Calculate overall benchmark score
def calculate_eq_bench_score(run_index, results, results_path, fullscale=False):
	global is_writing
	# We calculate an overall score for first pass answers and revised answers separately.
	# The final score is the best of these two numbers.

	if fullscale: # v2
		scores_key = 'individual_scores_fullscale'
	else: # v1 (normalised)
		scores_key = 'individual_scores'

	score_tally = 0	
	parseable_tally = 0
	n_iterations = results[run_index]['run_metadata']['total_iterations']
	n_iterations_tallied = 0

	for run_iter in results[run_index]['iterations']:
		if n_iterations_tallied >= n_iterations:
			break
		score_sum_first_pass = 0
		score_sum_revised = 0
		first_pass_parseable = 0
		revised_parseable = 0

		if not scores_key in results[run_index]['iterations'][run_iter]:
			continue

		for dialogue_id, r in results[run_index]['iterations'][run_iter][scores_key].items():

			r = results[run_index]['iterations'][run_iter][scores_key][dialogue_id]
			if 'first_pass_score' in r and r['first_pass_score'] != None:
				score_sum_first_pass += r['first_pass_score']
				first_pass_parseable += 1
			if 'revised_score' in r and r['revised_score'] != None:
				score_sum_revised += r['revised_score']
				revised_parseable += 1

		if first_pass_parseable:
			score_first_pass = 100 * (score_sum_first_pass / first_pass_parseable / 10)
		else:
			score_first_pass = 0
		
		if revised_parseable:
			score_revised = 100 * (score_sum_revised / revised_parseable / 10)
		else:
			score_revised = 0

		# If either the first pass or revised score has significantly less parseable answers,
		# we take the score with the higher number of parseable answers regardless of score.
		if score_revised >= score_first_pass and revised_parseable >= 0.95 * first_pass_parseable:
			final_score = score_revised
			final_parseable = revised_parseable			
		else:
			final_score = score_first_pass
			final_parseable = first_pass_parseable
			
		
		score_tally += final_score
		parseable_tally += final_parseable

		results_key = 'benchmark_results'
		if fullscale:
			results_key = 'benchmark_results_fullscale'
		results[run_index]['iterations'][run_iter][results_key] = {
			'first_pass_score': score_first_pass,
			'first_pass_parseable': first_pass_parseable,
			'revised_score': score_revised,
			'revised_parseable': revised_parseable,
			'final_score': final_score,
			'final_parseable': final_parseable
		}

		n_iterations_tallied += 1

	averaged_score = score_tally / n_iterations

	if fullscale:
		averaged_score = round(averaged_score, 2)	
	else:
		averaged_score = round(averaged_score, 2)

	safe_dump(results, results_path, max_retries=3)


	return (averaged_score, round(parseable_tally / n_iterations, 2))


neg_criteria = [
					"melodramatic", "shallow resolution", "unearned resolution",
					"simplistic moralizing", "shallow optimism", "forced optimism",
					"trite", "overwrought", "amateurish", "contrived", "uninspiring",
					"characters are too good", "incongruent ending positivity",
					"unearned transformations", "profundity over-reach", "amateurish descriptives",
					"clunky asides and interruptive sentence structures", "stilted dialogue",
					"tit-for-tat dialogue", "purple prose", "unsurprising or uncreative", "tell-don't-show",
					"weak dialogue", "meandering"
			]

def calculate_creative_writing_score(run_index, results, results_path):
	RELATIVE_SCORING = False
	prompt_scores = []  # List to hold total scores for each prompt
	iteration_averages = []  # To hold the average scores of the best half of each iteration
	
	for run_iter in results[run_index]['iterations']:
		prompt_scores = []
		for prompt_id, scores in results[run_index]['iterations'][run_iter]['individual_scores'].items():
			scoresum = 0
			
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
			if len(scores):
				prompt_scores.append(scoresum / len(scores))

		if len(prompt_scores) > 10:
			iteration_average = sum(prompt_scores) / len(prompt_scores)
			iteration_averages.append(iteration_average)

	# Average of iteration averages
	if iteration_averages:
		creative_writing_averaged_score = sum(iteration_averages) / len(iteration_averages)
	else:
		creative_writing_averaged_score = 0

	return round(10 * creative_writing_averaged_score, 2)


def calculate_creative_writing_score_judgemark(run_index, model_name, results):	
	RELATIVE_SCORING = False	
	iteration_averages = []  # To hold the average scores of the best half of each iteration
	raw_criteria_scores = []
	individual_item_scores = []

	for run_iter in results[run_index]['iterations']:		
		if int(run_iter) != 1:
			# limit judgemark to 1 iteration for now
			continue
		prompt_scores = []
		for prompt_id, scores in results[run_index]['iterations'][run_iter]['judgemark_results'][model_name]['individual_scores'].items():
			scoresum = 0
			for criteria, score in scores.items():
					criteria_lower = criteria.lower().strip()
					if RELATIVE_SCORING:						
						if any(neg_criterion in criteria_lower for neg_criterion in neg_criteria):
							this_criteria_score = ((-1 * score) + 10) / 2
						else:
							this_criteria_score = (score + 10) / 2
					else:
						if any(neg_criterion in criteria_lower for neg_criterion in neg_criteria):
							this_criteria_score = 10 - score
						else:
							this_criteria_score = score
					scoresum += this_criteria_score
					raw_criteria_scores.append(10*this_criteria_score)
			if len(scores):
				prompt_scores.append(scoresum / len(scores))
				individual_item_scores.append(10*scoresum / len(scores))

		if len(prompt_scores) > 10:
			iteration_average = sum(prompt_scores) / len(prompt_scores)
			iteration_averages.append(iteration_average)

	# Average of iteration averages
	if iteration_averages:
		creative_writing_averaged_score = sum(iteration_averages) / len(iteration_averages)
	else:
		creative_writing_averaged_score = 0

	return round(10 * creative_writing_averaged_score, 2), raw_criteria_scores, individual_item_scores
