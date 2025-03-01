import re
import os
from lib.run_query import run_query

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



# This is an undocumented feature. It will run a series of test prompts for
# each model and log the output.
def run_test_prompts(model, ooba_instance, 
							inference_engine, results, 
							 model_path, prompt_type, 
							 tokenizer, launch_ooba, 
							 ooba_request_timeout,
							 run_index, run_iter,
							 verbose):
	
	if 'test_prompts_results' in results[run_index]['iterations'][run_iter]:
		return
	if not os.path.exists(os.path.abspath('./test_prompts.txt')):
		return
	if inference_engine == 'transformers':
		print('! Custom test prompts only support ooba or openai as the inference engine.')
		return
	
	results[run_index]['iterations'][run_iter]['test_prompts_results'] = {}
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
			results[run_index]['iterations'][run_iter]['test_prompts_results'][sequence_name] = []
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
					results[run_index]['iterations'][run_iter]['test_prompts_results'][sequence_name].append(p.strip())
					history.append({"role": "user", "content": p.strip()})
					results[run_index]['iterations'][run_iter]['test_prompts_results'][sequence_name].append(inference.strip())
					history.append({"role": "assistant", "content": inference.strip()})

			with open('test_prompts_results.txt', 'a') as f:
				out_str = '\n\n### ' + run_index + '\n\n'
				out_str += 'NAME=' + sequence_name + '\n'
				out_str += 'TEMP=' + str(temp) + '\n'
				out_str += 'COMPLETION_TOKENS=' + str(completion_tokens) + '\n\n'
				out_str += '---\n'
				out_str += '\n\n---\n\n'.join(results[run_index]['iterations'][run_iter]['test_prompts_results'][sequence_name])
				f.write(out_str)

	except Exception as e:
		print(e)
		print('! Failed to run test prompts.')
		
def remove_revision_instructions(prompt, language):
	# cut out the part of the prompt asking for a revised answer
	if language == 'en':
		prompt = prompt.replace(' Then critique your answer by thinking it through step by step. Finally, give your revised scores.', '')
		prompt = prompt.replace('First pass scores:\n', '')
		prompt = prompt[:prompt.find('Critique: <your critique here>')] + '\n' + prompt[prompt.find('[End of answer]'):]
		prompt += '\nYour answer:\n'
	elif language == 'de':
		# de translation has some varations that we have to account for.
		substring_to_match = "Geben Sie jeder dieser möglichen Emotionen"
		replacement_string = "Geben Sie jeder dieser möglichen Emotionen eine Punktzahl von 0-10 für die relative Intensität, die sie wahrscheinlich fühlen werden."
		start_index = prompt.find(substring_to_match)
		end_index = prompt.find('\n', start_index)		
		prompt = prompt[:start_index] + replacement_string + prompt[end_index:]
		prompt = prompt.replace('Erste Bewertung:\n', '')
		prompt = prompt[:prompt.find('Kritik: <Ihre Kritik hier>')] + '\n' + prompt[prompt.find('[Ende der Antwort]'):]
		prompt += '\nIhre Antwort:\n'
	elif language == 'pl':
		prompt = prompt.replace(' Następnie skrytykuj swoją odpowiedź, przemyśl ją krok po kroku. Wprowadź zmiany i na koniec podaj ostateczne oceny.', '')
		prompt = prompt.replace('Pierwsze oceny:\n', '')
		prompt = prompt[:prompt.find('Weryfikacja: <twoja opinia tutaj>')] + '\n' + prompt[prompt.find('[Koniec odpowiedzi]'):]
		prompt += '\nTwoja odpowiedź:\n'
	return prompt