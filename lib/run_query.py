from transformers import pipeline
import time
import yaml
import requests

def run_chat_query(prompt, completion_tokens, model, tokenizer, temp):
	response, history = model.chat(tokenizer, prompt, history=None, max_length=completion_tokens, do_sample=True)
	return response

def run_pipeline_query(prompt, completion_tokens, model, tokenizer, temp):
	text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=completion_tokens, do_sample=True, temperature=temp)
	output = text_gen(prompt)
	out_str = output[0]['generated_text']
	# Trim off the prompt
	trimmed_output = out_str[len(prompt):].strip()
	return trimmed_output

def run_generate_query(prompt, completion_tokens, model, tokenizer, temp):
	inputs = tokenizer(prompt, return_tensors="pt")
	outputs = model.generate(inputs.input_ids, max_new_tokens=completion_tokens, do_sample=True, temperature=temp)
	output = tokenizer.decode(outputs[0], skip_special_tokens=True)
	# Trim off the prompt
	trimmed_output = output[len(prompt):].strip()
	return trimmed_output

# IF you are using transformers as your inferencing engine
# AND your model requires an inferencing method other than the default of transformers pipeline
# THEN specify your model & inferencing function here:
OPENSOURCE_MODELS_INFERENCE_METHODS = {
	'mistralai/Mistral-7B-Instruct-v0.1': run_generate_query,
	'Qwen/Qwen-14B-Chat': run_chat_query
}

def run_ooba_query(prompt, history, prompt_format, completion_tokens, temp, ooba_instance, launch_ooba, ooba_request_timeout):
	if launch_ooba and (not ooba_instance or not ooba_instance.url):
		raise Exception("Error: Ooba api not initialised")
	if launch_ooba:
		ooba_url = ooba_instance.url
	else:
		ooba_url = "http://127.0.0.1:5000"

	try:
		messages = history + [{"role": "user", "content": prompt}]
		data = {
        	"mode": "instruct",        
        	"messages": messages,
		  	"instruction_template": prompt_format,
		  	"max_tokens": completion_tokens,
    		"temperature": temp,
		}

		headers = {
			"Content-Type": "application/json"
		}		

		try:
			response = requests.post(ooba_url + '/v1/chat/completions', headers=headers, json=data, verify=False, timeout=ooba_request_timeout)			
			response = response.json()
		except Exception as e:
			print(e)
			# Sometimes the ooba api stops responding. If this happens we will get a timeout exception.
			# In this case we will try to restart ooba & reload the model.
			if launch_ooba:
				print('! Request failed to Oobabooga api. Attempting to reload Ooba & model...')
				ooba_instance.restart()				
		
		content = response['choices'][0]['message']['content']
		if content:
			return content.strip()
		else:
			print('Error: message is empty')
	except KeyboardInterrupt:
		print("Operation cancelled by user.")
		raise  # Re-raising the KeyboardInterrupt exception
	except Exception as e:
		print("Request failed.")
		print(e)
	return None


def run_openai_query(prompt, history, completion_tokens, temp, model, openai_client):
	try:
		messages = history + [{"role": "user", "content": prompt}]
		
		if model in OPENAI_COMPLETION_MODELS and openai_client.base_url == 'https://api.openai.com/v1/':
			result = openai_client.completions.create(
					model=model,
					temperature=temp,
					max_tokens=completion_tokens,
					prompt=prompt,
			)
			content = result.choices[0].text
		else: # assume it's a chat model
			result = openai_client.chat.completions.create(
					model=model,
					temperature=temp,
					max_tokens=completion_tokens,
					messages=messages,
			)
			content = result.choices[0].message.content

		if content:
			return content.strip()
		else:
			print('Error: message is empty')
			time.sleep(5)

	except Exception as e:
		print("Request failed.")
		print(e)
		time.sleep(5)

	return None

def run_poe_query(prompt, model, api_key_poe):
    try:

        import asyncio
        import fastapi_poe as fp
        import logging

        logging.basicConfig(level=logging.INFO, filename='poe.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

        # Define the asynchronous function to fetch responses
        async def get_responses(api_key_poe, messages):
            complete_response = ''
            async for partial in fp.get_bot_response(messages=messages, bot_name=model, api_key=api_key_poe):
                complete_response += partial.text  # Append each partial response
                #print (partial.text)
                logging.debug(f"Received partial response: {partial.text}")
                await asyncio.sleep(0.2)  # Minimal delay to stabilize timing
            await asyncio.sleep(4)
            return complete_response  # Return the complete response at the end

        # Prepare the message
        message = fp.ProtocolMessage(role="user", content=prompt.strip())

        # wrap the asynchronous call for synchronous use
        def sync_run():
            result = asyncio.run(get_responses(api_key_poe, [message]))
            return result

        # Execute the synchronous wrapper function
        response = sync_run()

        if response:
            print("Poe response:", response.strip())
            return response.strip()
        else:
            print('Error: message is empty')
            time.sleep(5)

    except Exception as e:
        print("Poe request failed.")
        print(e)
        time.sleep(5)

    return None


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

def parse_yaml(template_path):
	try:
		with open(template_path, 'r') as file:
			data = yaml.safe_load(file)
			# If the data is a string, replace \\n with \n
			if isinstance(data, str):
					data = data.replace('\\n', '\n')
			# If data is a dictionary, replace \\n in all string values
			elif isinstance(data, dict):
					for key, value in data.items():
						if isinstance(value, str):
							data[key] = value.replace('\\n', '\n')
			return data
	except FileNotFoundError:
		raise FileNotFoundError(f"Template file not found: {template_path}")
	
def generate_prompt_from_template(prompt, prompt_type):
	template_path = f"instruction-templates/{prompt_type}.yaml"
	template = parse_yaml(template_path)
	default_system_message = "You are an expert in emotional analysis."
	
	context = template["context"]
	if '<|system-message|>' in template['context']:
		if "system_message" in template and template["system_message"].strip():			
			context = context.replace("<|system-message|>", template["system_message"])
		else:
			context = context.replace("<|system-message|>", default_system_message)

	turn_template = template["turn_template"].replace("<|user|>", template["user"]).replace("<|bot|>", template["bot"])	
	formatted_prompt = context + turn_template
	formatted_prompt = formatted_prompt.split("<|bot-message|>")[0]
	return formatted_prompt.replace("<|user-message|>", prompt)

def run_query(model_path, prompt_format, prompt, history, completion_tokens, model, tokenizer, temp, inference_engine, ooba_instance, launch_ooba, ooba_request_timeout, openai_client, api_key_poe):
	if inference_engine == 'openai':
		return run_openai_query(prompt, history, completion_tokens, temp, model_path, openai_client)
	elif inference_engine == 'poe':
		return run_poe_query(prompt, model_path, api_key_poe)
	elif inference_engine == 'ooba':
		return run_ooba_query(prompt, history, prompt_format, completion_tokens, temp, ooba_instance, launch_ooba, ooba_request_timeout)
	else: # transformers
		# figure out the correct inference method to use
		if model_path in OPENSOURCE_MODELS_INFERENCE_METHODS:
			inference_fn = OPENSOURCE_MODELS_INFERENCE_METHODS[model_path]
		else:
			inference_fn = run_pipeline_query

		formatted_prompt = generate_prompt_from_template(prompt, prompt_format)		
		return inference_fn(formatted_prompt, completion_tokens, model, tokenizer, temp)


