from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel

def load_model(base_model_path, lora_path, quantization):
	tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

	# This is for llama2 models, but doesn't seem to have
	# adverse effects on benchmarks for other models.
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "right"

	# Quantization Config
	if quantization == '4bit':
		# load as 4 bit
		quant_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.float16,
			bnb_4bit_use_double_quant=False
		)
	elif quantization == '8bit':
		# load as 8 bit
		quant_config = BitsAndBytesConfig(
			load_in_8bit=True,		
		)
	else:
		quant_config = None

	# Model
	if quant_config:
		base_model = AutoModelForCausalLM.from_pretrained(
			base_model_path,
			quantization_config=quant_config,
			device_map={"": 0},
			trust_remote_code=True
		)
	else:
		base_model = AutoModelForCausalLM.from_pretrained(
			base_model_path,
			device_map={"": 0},
			trust_remote_code=True
		)

	if lora_path:
		peft_model = PeftModel.from_pretrained(base_model, lora_path)
		return peft_model, tokenizer
	else:
		return base_model, tokenizer
	
def load_model_01ai(base_model_path, lora_path, quantization):
	# Quantization Config
	if quantization == '4bit':
		# load as 4 bit
		quant_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.float16,
			bnb_4bit_use_double_quant=False
		)
	elif quantization == '8bit':
		# load as 8 bit
		quant_config = BitsAndBytesConfig(
			load_in_8bit=True,		
		)
	else:
		quant_config = None

	base_model = AutoModelForCausalLM.from_pretrained(base_model_path, quantization_config=quant_config, device_map="auto", torch_dtype="auto", trust_remote_code=True)
	tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

	if lora_path:
		peft_model = PeftModel.from_pretrained(base_model, lora_path)
		return peft_model, tokenizer
	else:
		return base_model, tokenizer

	