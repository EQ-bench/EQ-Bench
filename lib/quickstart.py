import subprocess
import os
import sys
# Add the root directory to the system path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import lib.ooba

def run_command(command, cwd=None, shell=True):
	process = subprocess.Popen(command, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd=cwd)
	while True:
		output = process.stdout.readline()
		print(output, end='')
		if output == '' and process.poll() is not None:
			break
	return process.poll()

# Step 1: Clone repo and run start_linux.sh
if not os.path.exists(os.path.expanduser('~/text-generation-webui')):
	print("Cloning text-generation-webui and running start_linux.sh")
	run_command("git clone https://github.com/oobabooga/text-generation-webui.git", cwd=os.path.expanduser("~"))

# Step 2: Monitor start_linux.sh output
print("Running start_linux.sh...")
#run_script_and_monitor_output("./start_linux.sh", "Starting Text generation web UI", cwd=os.path.expanduser("~/text-generation-webui"))
ooba_instance = lib.ooba.Ooba(os.path.expanduser("~/text-generation-webui") + '/start_linux.sh', None, None, True, load_model=False, automate_prompts=True)
ooba_started_ok = ooba_instance.start()
if not ooba_started_ok:
	print('Ooba failed to launch.')
else:
	print('Ooba installed & launched successfully.')
try:
	print('Closing ooba...')
	ooba_instance.stop()
except Exception as e:
	print(e)

# Step 3: Go to EQ-Bench, make install_reqs.sh executable and run it
print("Running install_reqs.sh in EQ-Bench")
run_command("chmod +x install_reqs.sh", cwd=os.path.abspath("./"))
run_command("./install_reqs.sh", cwd=os.path.abspath("./"))

# Step 4: Continue with remaining commands

#if ooba_started_ok:
#	print("Running eq-bench.py")
#	run_command("python eq-bench.py -v -f", cwd=os.path.abspath("./"))