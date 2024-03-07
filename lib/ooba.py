import time
import subprocess
import threading
import re
import psutil
import os
import sys
import shlex
import queue
import platform
from lib.download import download_model

class Ooba:
	def __init__(self, script_path, model_path, cache_dir=None, verbose=False, trust_remote_code=False, ooba_args_global="", ooba_args="", fast_download=False, include_patterns=None, exclude_patterns=None, hf_access_token=None, load_model=True):
		self.script_path = script_path
		if script_path.endswith('sh'):
			self.script_command = 'bash'
		elif script_path.endswith('server.py'):
			self.script_command = sys.executable
		elif script_path.endswith('bat'):
			self.script_command = 'powershell'
		self.ooba_dir = os.path.dirname(os.path.abspath(self.script_path))
		self.model_path = model_path
		self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir)) if cache_dir else None
		self.verbose = verbose
		self.trust_remote_code = trust_remote_code
		self.ooba_args_global = ooba_args_global
		self.ooba_args = ooba_args
		self.fast_download = fast_download
		self.include_patterns = include_patterns
		self.exclude_patterns = exclude_patterns
		self.hf_access_token = hf_access_token
		self.load_model = load_model

		self.process = None
		self.url_found_event = threading.Event()
		self.shutdown_message_shown = threading.Event()
		self.process_end_event = threading.Event()
		self.output_queue = queue.Queue()

	def is_already_running(self):
		script_dir = os.path.dirname(os.path.abspath(self.script_path))
		for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
			try:
					if 'python' in proc.name().lower() and any('server.py' in cmd for cmd in proc.info['cmdline']) and proc.info['cwd'] == script_dir:
						return True
			except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
					continue
		return False

	def download_model(self):
		# figure out if this is an existing local path
		if os.path.exists(self.model_path):			
			self.command_args = [self.script_command, self.script_path, '--model', self.model_path]
		else:			
			# it's not an existing local path, so try using hf_hub downloader to fetch the model
			download_path = download_model(self.model_path, self.cache_dir, self.ooba_dir, self.include_patterns, self.exclude_patterns, self.hf_access_token)

			# Check if the download path was found
			if download_path is None:
				raise Exception("Download path not found in the output.")
			else:
				print(f"Model downloaded to: {download_path}")

			self.model_downloaded_fullpath = download_path

			model_dir, model = os.path.split(download_path)

			self.command_args = [self.script_command, self.script_path, '--model', model, '--model-dir', model_dir]
			
		
			return

	def build_command(self):
		if self.load_model:
			self.download_model()
		else:
			# this path is used by quickstart.py when first launching ooba
			self.command_args = [self.script_command, self.script_path]

		if self.trust_remote_code:
			self.command_args += ['--trust-remote-code']		
		
		self.command_args += ['--api']

		if self.ooba_args:
			self.command_args += shlex.split(self.ooba_args)
		elif self.ooba_args_global:
			self.command_args += shlex.split(self.ooba_args_global)
		return self.command_args

	def start(self):
		if self.is_already_running():
			print("Ooba server is already running.")
			return

		command = self.build_command()
		if self.verbose:
			print('Launching ooba with command:')
			print(' '.join(command))
		
		# this is to reduce buffering of ooba's output so we can capture it properly
		if platform.system() != 'Windows':
			# Disable output buffering on Unix-like systems (Linux, macOS)
			#command = ['stdbuf', '-o0', '-e0'] + command # unreliable
			#command = ['unbuffer', '-p'] + command # requires 'expect' package to be installed with apt-get
			command_string = " ".join(shlex.quote(arg) for arg in command)
			command = ['script', '-q', '/dev/null', '-c', command_string]
			
		self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, cwd=os.path.dirname(self.script_path))

		self.stdout_thread = threading.Thread(target=self.monitor_output, args=('stdout',), daemon=True).start()
		self.stderr_thread = threading.Thread(target=self.monitor_output, args=('stderr',), daemon=True).start()
		self.stdin_thread = threading.Thread(target=self.send_user_input, daemon=True).start()
		
		self.url_found_event.wait()  # This will block until the event is set

		if self.url_found_event.is_set():
			print(f"URL found: {self.url}")
			return self.url
		else:
			print("Process ended before URL was found.")
			return None

	def send_user_input(self):
		while True:
			try:
				user_input = input()
				if self.process:
					print('sending',user_input,'to subprocess')
					self.process.stdin.write(user_input + "\n")
					self.process.stdin.flush()
			except (EOFError, KeyboardInterrupt):
				break

	def monitor_output(self, pipe):
		url_pattern1 = re.compile(r"http://127.0.0.1:5000")
		url_pattern2 = re.compile(r"Running on local URL")
		error_pathnotfound_pattern = re.compile(r"The path to the model does not exist\. Exiting\.")
		shutdown_pattern = re.compile(r"Shutting down Text generation web UI gracefully\.")
		exception_pattern = re.compile(r"Traceback \(most recent call last\)")

		while True:
			#line = ''
			if pipe == 'stdout':
				line = self.process.stdout.readline()
			elif pipe == 'stderr':
				line = self.process.stderr.readline()
			if not line:  # EOF
				self.process_end_event.set()
				break
			if line.endswith('\n') or pipe == 'stderr':
				print(line, end='')

			if platform.system() == 'Windows':
				if url_pattern1.search(line):
					# There are buffering issues with ooba for the "running on local URL" line
					# We can't solve this with stdbuf since it doesn't exist on windows. so we
					# are taking the hacky approach and waiting for the prior line and pausing for 5s.
					self.url = 'http://127.0.0.1:5000'
					time.sleep(5)
					self.url_found_event.set()
					continue
			else:
				if url_pattern2.search(line):
					self.url = 'http://127.0.0.1:5000'
					time.sleep(5)
					self.url_found_event.set()
					continue

			if error_pathnotfound_pattern.search(line) or exception_pattern.search(line):
				print("Error detected in process output.")					
				self.stop(force_exit=True)
				break

			elif shutdown_pattern.search(line):
				print("Shutdown signal detected.")
				self.shutdown_message_shown.set()
				break

	def wait_for_url_or_process_end(self):
		while not self.url_found_event.is_set() and not self.process_end_event.is_set():
			time.sleep(0.1)
		if self.url_found_event.is_set():
			print("URL found.")
		else:
			print("Process ended before URL was found.")

	def restart(self):
		self.stop()
		self.process_end_event.clear()
		self.shutdown_message_shown.clear()
		self.url_found_event.clear()
		self.output_lock = threading.Lock()
		self.process = None
		start_result = self.start()
		return start_result
	
	def stop(self, force_exit=False, timeout=5):
		self.process.terminate()
		try:
			self.process.wait(timeout)
		except Exception:
			self.process.kill()
		self.process_end_event.set()
		return

