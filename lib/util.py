import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
import os
import psutil
import shutil

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

def parse_downloader_args(downloader_args_str):
	pattern = re.compile(r'--(include|exclude) (\[.*?\]|".*?"|\'.*?\')')
	include_patterns = []
	exclude_patterns = []

	for match in pattern.finditer(downloader_args_str):
		arg_type, arg_value = match.groups()

		# Removing surrounding brackets or quotes
		cleaned_arg_value = re.sub(r'^\[|\]$|\'|"', '', arg_value)

		# Splitting if it's a list, considering both single and double quotes
		if ',' in cleaned_arg_value:
			values = re.findall(r'\'(.*?)\'|"([^"]*)"|([^\s,]+)', cleaned_arg_value)
			# Flatten the list of tuples and filter out empty strings
			values = [item for sublist in values for item in sublist if item]
		else:
			values = [cleaned_arg_value.strip()]

		# Validation for non-empty strings
		values = [v for v in values if v]

		if not values:
			raise ValueError(f"Invalid or empty pattern found for --{arg_type} argument.")

		if arg_type == 'include':
			include_patterns.extend(values)
		elif arg_type == 'exclude':
			exclude_patterns.extend(values)

	return include_patterns, exclude_patterns

def parse_batch(batch, ooba_launch_script, launch_ooba):
	parsed = []
	downloader_args_pattern = re.compile(r'(--include \[.*?\]|--include ".*?"|--exclude \[.*?\]|--exclude ".*?")')
	
	for line in batch:
		line = line.strip()
		line_orig = line
		if not line or line.startswith('#'):
			continue

		try:
			# Extract downloader_args using regex
			downloader_args = downloader_args_pattern.findall(line)
			downloader_args_str = ' '.join(downloader_args)
			include_patterns, exclude_patterns = parse_downloader_args(downloader_args_str)
			# Remove downloader_args from the line
			line = downloader_args_pattern.sub('', line).strip(',')

			# Parse out ooba params
			parts = line.split(',')
			ooba_params = parts[7:-1]
			ooba_params_str = ','.join(ooba_params)
			if ooba_params_str.strip():
				# Remove the ooba params from the string before continuing (as it may contain commas)
				line = line[:line.rfind(ooba_params_str)] + line[line.rfind(ooba_params_str) + len(ooba_params_str):] if ooba_params_str in line else line

			# Parse and validate individual fields
			run_id, prompt_format, model_path, lora_path, quantization, n_iterations, inference_engine = parts[:7]

			run_id = run_id.strip()
			if not run_id:
				raise Exception('Missing run id.')
			model_path = model_path.strip()			
			if not model_path and launch_ooba:
				raise Exception('Missing model path.')
			if model_path:
				# Note: if a hf model id starts with '~', that will cause issues here.
				if model_path.startswith('~'):
					model_path = os.path.expanduser(model_path)
				if os.path.exists(model_path):
					model_path = os.path.abspath(model_path)

			lora_path = lora_path.strip()
			if lora_path:
				if lora_path.startswith('~'):
					lora_path = os.path.expanduser(lora_path)
				lora_path = os.path.abspath(lora_path)
			
			quantization = quantization.strip().lower()
			if quantization not in QUANT_TYPES:
				raise Exception('Error: invalid quantization type. Check config. ' + quantization)
			n_iterations = n_iterations.strip()
			if not is_int(n_iterations) or int(n_iterations) <= 0:
				raise Exception('Invalid number of repeats. Must be an integer > 0. ' + n_iterations)
			
			# Read inference engine option from config
			inference_engine = inference_engine.strip().lower()
			if inference_engine not in ['transformers', 'ooba', 'openai']:
				raise Exception("inference_engine in config.cfg must be transformers, openai or oobabooga.")
			if inference_engine == 'ooba' and not ooba_launch_script:
				raise Exception('ooba_launch_script not set in config.cfg')
			
			prompt_format = prompt_format.strip()
			if inference_engine == 'transformers':
				template_path = './instruction-templates/' + prompt_format + '.yaml'
				if (not prompt_format) or not os.path.exists(template_path):
					raise Exception('Error: prompt template not found: ' + template_path)
			elif inference_engine == 'ooba':
				ooba_dir = os.path.dirname(ooba_launch_script)
				template_path = ooba_dir + '/instruction-templates/' + prompt_format + '.yaml'
				if (not prompt_format) or not os.path.exists(template_path):
					raise Exception('Error: prompt template not found: ' + template_path)

			parsed.append((
					run_id, prompt_format, model_path, lora_path, quantization, int(n_iterations), inference_engine, ooba_params_str, include_patterns, exclude_patterns
			))

		except Exception as e:
			print(e)
			print('Failed to parse line in config:')
			print(line_orig)
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

def get_process_pwd(pid):
	try:
		process = psutil.Process(pid)
		return process.cwd()
	except (psutil.NoSuchProcess, psutil.AccessDenied):
		print(f"Cannot access process with PID {pid}")
		return None
	
def read_output(master_fd, output_lines):
	buffer = ''
	try:
		while True:
			output = os.read(master_fd, 1024)
			if not output:
					break
			buffer += output.decode()
			while '\n' in buffer:
					line, buffer = buffer.split('\n', 1)
					output_lines.append(line)
					print(line)
	except KeyboardInterrupt:
		print("Process interrupted")
	except Exception as e:
		# process is probably dead
		return

# Huggingface cache uses symlinks in the model snapshot dir, so we need to iterate
# over each file and delete the source symlinked files first.
def delete_symlinks_and_dir(dir_to_delete, verbose):
	# Check if the directory exists
	if not os.path.exists(dir_to_delete):
		print(f"Directory {dir_to_delete} does not exist.")
		return

	# Iterate through the items in the directory
	for item in os.listdir(dir_to_delete):
		item_path = os.path.join(dir_to_delete, item)

		# Check if the item is a symlink
		if os.path.islink(item_path):
			source_path = os.readlink(item_path)

			# Resolve the source path relative to the symlink's directory
			if not os.path.isabs(source_path):
					source_path = os.path.join(os.path.dirname(item_path), source_path)

			# Check if the source file exists and is not a directory
			if os.path.exists(source_path) and not os.path.isdir(source_path):
					if verbose:
						print(f"Deleting source file of symlink: {source_path}")
					os.remove(source_path)
			else:
					print(f"Source file does not exist or is a directory: {source_path}")

	# Delete the directory and its contents
	shutil.rmtree(dir_to_delete)
	if verbose:
		print(f"Deleted directory: {dir_to_delete}")