import os
from huggingface_hub import snapshot_download

def download_model(model_path, cache_dir, ooba_dir, include, exclude, hf_access_token):
	print('Downloading model', model_path)

	# if no include filters were specified, include all by default
	if not include:
		include = ["*"]

	if cache_dir:
		cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
		models_dir = cache_dir
	else:
		models_dir = "~/.cache/huggingface/hub"
		models_dir = os.path.abspath(os.path.expanduser(models_dir))

	try:
		this_snapshot_path = snapshot_download(repo_id=model_path, repo_type="model", allow_patterns=include, ignore_patterns=exclude, cache_dir=cache_dir, token=hf_access_token, resume_download=True)
	except Exception as e:
		print('! Error downloading model')
		raise(e)

	return this_snapshot_path