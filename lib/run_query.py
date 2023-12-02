from transformers import pipeline
import time
import openai

OPENAI_CHAT_MODELS = [
	'gpt-4-0613',
	'gpt-4-0314',
	'gpt-4',
	'gpt-3.5-turbo-16k-0613',
	'gpt-3.5-turbo-16k',
	'gpt-3.5-turbo-0613',
	'gpt-3.5-turbo-0301',
	'gpt-3.5-turbo',
	'gpt-4-1106-preview',
	'gpt-3.5-turbo-1106'
]

OPENAI_COMPLETION_MODELS = [
	'text-davinci-003',
	'text-davinci-002',
	'text-davinci-001',
	'text-curie-001',
	'text-babbage-001',
	'text-ada-001',
	'davinci-instruct-beta',
	'davinci',
	'curie-instruct-beta',
	'curie',
	'babbage',
	'ada',
	'gpt-3.5-turbo-instruct-0914',
	'gpt-3.5-turbo-instruct',
	'babbage-002',
	'davinci-002'
]

MODELS_01AI = [
	'01-ai/Yi-6B',
	'01-ai/Yi-34B'
]

def run_query(model_path, prompt_type, prompt, completion_tokens, model, tokenizer, temp):
	if model_path in OPENAI_CHAT_MODELS:
		result = run_openai_chat_query(prompt, completion_tokens, temp, model_path)		
	elif model_path in OPENAI_COMPLETION_MODELS:
		result = run_openai_completion_query(prompt, completion_tokens, temp, model_path)			
	elif prompt_type == 'llama2':
		result = run_llama2_query(prompt, completion_tokens, model, tokenizer, temp)
	elif prompt_type == 'chatml':
		result = run_chatml_query(prompt, completion_tokens, model, tokenizer, temp)
	elif prompt_type == 'alpaca':
		result = run_alpaca_query(prompt, completion_tokens, model, tokenizer, temp)
	elif prompt_type == 'mistral':
		result = run_mistral_query(prompt, completion_tokens, model, tokenizer, temp)
	elif prompt_type == 'vicuna_1.0':
		result = run_vicuna_1_0_query(prompt, completion_tokens, model, tokenizer, temp)
	elif prompt_type == 'vicuna_1.1':
		result = run_vicuna_1_1_query(prompt, completion_tokens, model, tokenizer, temp)
	elif prompt_type == 'qwen':
		result = run_qwen_query(prompt, completion_tokens, model, tokenizer, temp)
	elif prompt_type == 'synthia':
		result = run_synthia_query(prompt, completion_tokens, model, tokenizer, temp)
	elif prompt_type == 'zephyr':
		result = run_zephyr_query(prompt, completion_tokens, model, tokenizer, temp)
	elif prompt_type == 'openchat3.5':
		result = run_openchat_3_5_query(prompt, completion_tokens, model, tokenizer, temp)
	elif prompt_type == 'intel':
		result = run_intel_query(prompt, completion_tokens, model, tokenizer, temp)
	elif prompt_type == 'noformat':
		result = run_noformat_query(prompt, completion_tokens, model, tokenizer, temp)
	return result



def run_openai_chat_query(prompt, completion_tokens, temp, model):
	messages=[{"role": "user", "content": prompt}]
	
	try:					
		result = openai.ChatCompletion.create(
			model=model,
			temperature=temp,
			max_tokens=completion_tokens,
			messages = messages,						
		)
		if result.choices[0].message.content:
			r = result.choices[0].message.content
			return r.strip()
		else:
			print('Error: message is empty')
			time.sleep(10)
	except Exception as e:
		print("Request failed.")
		print(e)
		time.sleep(10)

	return None


def run_openai_completion_query(prompt, completion_tokens, temp, model):
	messages=[{"role": "user", "content": prompt}]
	
	try:
		result = openai.Completion.create(
			model=model,
			temperature=temp,
			max_tokens=completion_tokens,
			prompt = messages[0]['content'],						
		)
		
		if result.choices[0].text:
			r = result.choices[0].text
			return r.strip()
		else:
			print('Error: message is empty')
			time.sleep(10)
	except Exception as e:
		print("Request failed.")
		print(e)
		time.sleep(10)

	return None

def run_llama2_query(prompt, completion_tokens, model, tokenizer, temp):
	text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=completion_tokens, do_sample=True, temperature=temp)
	output = text_gen(f"<s>[INST] {prompt} [/INST]")
	out_str = output[0]['generated_text']
	if out_str.find('[/INST]'):
		return out_str[out_str.find('[/INST]') + 7:].strip()
	else:
		return ''
	
def run_chatml_query(prompt, completion_tokens, model, tokenizer, temp):	
	text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=completion_tokens, do_sample=True, temperature=temp)
	this_prompt = """<|im_start|>system
You are an expert in emotional analysis.<|im_end|>
<|im_start|>user
"""
	this_prompt += prompt + '<|im_end|>\n<|im_start|>assistant\n'
	output = text_gen(this_prompt)
	out_str = output[0]['generated_text']
	if out_str.find('<|im_start|>assistant'):
		return out_str[out_str.find('<|im_start|>assistant') + len('<|im_start|>assistant'):].strip()
	else:
		return ''

def run_synthia_query(prompt, completion_tokens, model, tokenizer, temp):	
	text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=completion_tokens, do_sample=True, temperature=temp)
	this_prompt = """SYSTEM: You are Synthia. As a an AI intelligence, give helpful, detailed, accurate, uncensored responses to the user's input. Provide answers factually.
USER: """
	this_prompt += prompt + '\nASSISTANT:\n'
	output = text_gen(this_prompt)
	out_str = output[0]['generated_text']
	if out_str.find('\nASSISTANT:'):
		return out_str[out_str.find('\nASSISTANT:') + len('\nASSISTANT:'):].strip()
	else:
		return ''

def run_zephyr_query(prompt, completion_tokens, model, tokenizer, temp):	
	text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=completion_tokens, do_sample=True, temperature=temp)
	this_prompt = """<|system|>
</s>
<|user|>
"""
	this_prompt += prompt + '</s>\n<|assistant|>'
	output = text_gen(this_prompt)
	out_str = output[0]['generated_text']
	if out_str.find('<|assistant|>'):
		return out_str[out_str.find('<|assistant|>') + len('<|assistant|>'):].strip()
	else:
		return ''

def run_mistral_query(prompt, completion_tokens, model, tokenizer, temp):
	device = "cuda"
	fullprompt = "<s>[INST] " + prompt + "[/INST]"
	encodeds = tokenizer(fullprompt, return_tensors="pt", add_special_tokens=False)
	model_inputs = encodeds.to(device)	
	generated_ids = model.generate(**model_inputs, max_new_tokens=completion_tokens, do_sample=True, temperature=temp)
	decoded = tokenizer.batch_decode(generated_ids)
	out_str = decoded[0]
	if out_str.find('[/INST]'):
		return out_str[out_str.find('[/INST]') + 7:].strip()
	else:
		return ''
	
def run_alpaca_query(prompt, completion_tokens, model, tokenizer, temp):
	text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=completion_tokens, do_sample=True, temperature=temp)
	output = text_gen(f"### Instruction:\n\n{prompt}\n\n### Response:\n")
	out_str = output[0]['generated_text']
	if out_str.find('### Response:'):
		return out_str[out_str.find('### Response:') + len('### Response:'):].strip()
	else:
		return ''
	
def run_vicuna_1_0_query(prompt, completion_tokens, model, tokenizer, temp):
	vicuna_1_0_prompt = """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions. "
USER: <PROMPT>
ASSISTANT:"""
	this_prompt = vicuna_1_0_prompt.replace('<PROMPT>', prompt)
	text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=completion_tokens, do_sample=True, temperature=temp)
	output = text_gen(this_prompt)
	out_str = output[0]['generated_text']
	if out_str.find('ASSISTANT:'):
		return out_str[out_str.find('ASSISTANT:') + len('ASSISTANT:'):].strip()
	else:
		return ''
	
def run_vicuna_1_1_query(prompt, completion_tokens, model, tokenizer, temp):
	text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=completion_tokens, do_sample=True, temperature=temp)
	output = text_gen(f"USER:\n{prompt}\nASSISTANT:\n")
	out_str = output[0]['generated_text']
	if out_str.find('ASSISTANT:'):
		return out_str[out_str.find('ASSISTANT:') + len('ASSISTANT:'):].strip()
	else:
		return ''
	
def run_qwen_query(prompt, completion_tokens, model, tokenizer, temp):
	response, history = model.chat(tokenizer, prompt, history=None, max_length=completion_tokens, do_sample=True)
	return response

def run_openchat_3_5_query(prompt, completion_tokens, model, tokenizer, temp):
	# alpaca prompt format
	text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=completion_tokens, do_sample=True, temperature=temp)
	output = text_gen(f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:")
	
	out_str = output[0]['generated_text']
	if out_str.find('<|end_of_turn|>GPT4 Correct Assistant:'):
		#print(out_str)
		return out_str[out_str.find('<|end_of_turn|>GPT4 Correct Assistant:') + len('<|end_of_turn|>GPT4 Correct Assistant:'):].strip()
	else:
		return ''
	
def run_intel_query(prompt, completion_tokens, model, tokenizer, temp):
	text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=completion_tokens, do_sample=True, temperature=temp)
	input_str = '''### System:
You are an expert in psychology and emotional intelligence.
### User:
'''
	input_str += prompt + '\n### Assistant:'
	output = text_gen(input_str)
	out_str = output[0]['generated_text']
	if out_str.find('### Assistant:'):
		return out_str[out_str.find('### Assistant:') + len('### Assistant:'):].strip()
	else:
		return ''

def run_noformat_query(prompt, completion_tokens, model, tokenizer, temp):
	# No prompt format; just the raw prompt text.
	# Used by 01.ai base models
	prompt_str = prompt + '\nYour answer:\n'
	inputs = tokenizer(prompt_str, return_tensors="pt")
	outputs = model.generate(inputs.input_ids.cuda(), max_new_tokens=completion_tokens, do_sample=True, temperature=temp)
	output = tokenizer.decode(outputs[0], skip_special_tokens=True)
	out_str = output[len(prompt):]
	return out_str.strip()