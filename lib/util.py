import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
import os

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
def parse_batch(batch, ooba_launch_script):
	parsed = []
	for line in batch:
		line = line.strip()
		if not line or line.startswith('#'):
			continue
		try:
			run_id, prompt_format, model_path, lora_path, quantization, n_iterations, inference_engine, ooba_params = [x.strip() for x in line.split(',')]			
			run_id = run_id.strip()
			if not run_id:
				raise Exception('Missing run id.')
			model_path = model_path.strip()			
			if not model_path:
				raise Exception('Missing model path.')
			lora_path = lora_path.strip()
			prompt_format = prompt_format.strip()
			template_path = './instruction-templates/' + prompt_format + '.yaml'
			if (not prompt_format) or not os.path.exists(template_path):
				raise Exception('Error: prompt template not found: ' + template_path)
			quantization = quantization.strip().lower()
			if quantization not in QUANT_TYPES:
				raise Exception('Error: invalid quantization type. Check config. ' + quantization)
			n_iterations = n_iterations.strip()
			if not is_int(n_iterations) or int(n_iterations) <= 0:
				raise Exception('Invalid number of repeats. Must be an integer > 0. ' + n_iterations)
			
			# Read inference engine option from config
			inference_engine = inference_engine.lower()
			if inference_engine not in ['transformers', 'oobabooga', 'openai']:
				raise Exception("inference_engine in config.cfg must be transformers, openai or oobabooga.")
			if inference_engine == 'oobabooga' and not ooba_launch_script:
				raise Exception('ooba_launch_script not set in config.cfg')
						
			
			parsed.append((
				run_id, prompt_format, model_path, lora_path, quantization, int(n_iterations), inference_engine, ooba_params
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