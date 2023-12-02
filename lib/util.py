import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re

PROMPT_TYPES = [
	'openai_api',
	'chatml',
	'alpaca',
	'llama2',
	'mistral',
	'vicuna_1.0',
	'vicuna_1.1',
	'qwen',
	'synthia',
	'zephyr',
	'openchat3.5',
	'intel',
	'noformat'
]
QUANT_TYPES = [
	'8bit',
	'4bit',
	'none',
	''
]

def is_int(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

# Parse all the benchmark runs specified in the config file.
def parse_batch(batch):
	parsed = []
	for line in batch:
		line = line.strip()
		if not line or line.startswith('#'):
			continue
		try:
			run_id, prompt_type, model_path, lora_path, quantization, n_iterations = [x.strip() for x in line.split(',')]			
			run_id = run_id.strip()
			if not run_id:
				raise Exception('Missing run id.')
			model_path = model_path.strip()			
			if not model_path:
				raise Exception('Missing model path.')
			lora_path = lora_path.strip()
			prompt_type = prompt_type.strip().lower()
			if (not prompt_type) or prompt_type not in PROMPT_TYPES:
				raise Exception('Error: invalid prompt format ' + prompt_type)
			quantization = quantization.strip().lower()
			if quantization not in QUANT_TYPES:
				raise Exception('Error: invalid quantization type. Check config. ' + quantization)
			n_iterations = n_iterations.strip()
			if not is_int(n_iterations) or int(n_iterations) <= 0:
				raise Exception('Invalid number of repeats. Must be an integer > 0. ' + n_iterations)
			parsed.append((
				run_id, prompt_type, model_path, lora_path, quantization, int(n_iterations)
			))
			
		except Exception as e:
			print(e)
			print('Failed to parse line in config:')
			print(line)
			exit()
	return parsed



def upload_results_google_sheets(google_spreadsheet_url, result):
	match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', google_spreadsheet_url)
	sheet_id = match.group(1) if match else None

	if not sheet_id:
		print('! Error: failed to parse sheet id from url')
		return

	# Load credentials
	scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
	creds = ServiceAccountCredentials.from_json_keyfile_name('./google_creds.json', scope)
	client = gspread.authorize(creds)
	
	sheet = client.open_by_key(sheet_id)
	worksheet = sheet.get_worksheet(0)
	worksheet.append_row(result)