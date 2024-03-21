import firebase_admin
from firebase_admin import firestore
import time

firebase = None
db = None

def init_db():
	global firebase, db
	if not firebase_admin._apps:
		firebase = firebase_admin.initialize_app()
	db = firestore.client()
	print('Firebase db initialised:', firebase.name)

def save_eq_bench_result_to_db(results, score, parseable, last_error, run_index, bench_success):
	global db

	if not db:
		return
	
	if score:
		score = round(score, 2)
	
	try:
		meta = results['run_metadata']
		if meta['eq_bench_version'] == 'v1':
			n_questions_total = 60
		else:			
			n_questions_total = 171

		raw_results = {}

		for i in range(meta['total_iterations']):
			iter_index = str(i+1)
			if iter_index in results['iterations']:
				if meta['eq_bench_version'] == 'v1':
					individual_scores = results['iterations'][iter_index]['individual_scores']
				else:
					individual_scores = results['iterations'][iter_index]['individual_scores_fullscale']
				raw_results[iter_index] = {
					'respondent_answers': results['iterations'][iter_index]['respondent_answers'],
					'individual_scores': individual_scores,
					'raw_inference': results['iterations'][iter_index]['raw_inference']
				}

		to_save={
			'index_string': run_index,
			'run_id': meta['run_id'],
			'run_completed': int(time.time()),
			'benchmark_success': bench_success,
			'eqbench_version': meta['eq_bench_version'],
			'benchmark_type': 'eq-bench',
			'language': meta['language'],
			'n_questions_parseable': parseable,
			'n_questions_total': n_questions_total,
			'benchmark_score': score,
			'instruction_template': meta['instruction_template'],
			'model_path': meta['model_path'],
			'lora_path': meta['lora_path'],
			'bitsandbytes_quant': meta['bitsandbytes_quant'],
			'total_iterations': meta['total_iterations'],
			'inference_engine': meta['inference_engine'],
			'ooba_params': meta['ooba_params'],
			'include_patterns': meta['include_patterns'],
			'exclude_patterns': meta['exclude_patterns'],
			'errors': last_error,
			'raw_results': raw_results
		}

		db.collection("benchmark_results").add(to_save)
		print('Results saved to firebase db.')
	except Exception as e:
		print(e)
		print('! Failed to save results to db.')


def save_creative_writing_result_to_db(results, score, parseable, last_error, run_index, bench_success):
	global db

	if not db:
		return
	
	try:
		meta = results['run_metadata']

		raw_results = {}

		for i in range(meta['total_iterations']):
			iter_index = str(i+1)
			if iter_index in results['iterations']:
					raw_results[iter_index] = {
						'test_model_response': results['iterations'][iter_index]['test_model_response'],
						'judge_model_response': results['iterations'][iter_index]['judge_model_response'],
						'individual_scores': results['iterations'][iter_index]['individual_scores']
					}

		to_save = {
			'index_string': run_index,
			'run_id': meta['run_id'],
			'run_completed': int(time.time()),
			'benchmark_success': bench_success,
			'benchmark_type': 'creative-writing',
			'benchmark_score': score,
			'model_path': meta['model_path'],
			'lora_path': meta['lora_path'],
			'bitsandbytes_quant': meta['bitsandbytes_quant'],
			'total_iterations': meta['total_iterations'],
			'inference_engine': meta['inference_engine'],
			'ooba_params': meta['ooba_params'],
			'include_patterns': meta['include_patterns'],
			'exclude_patterns': meta['exclude_patterns'],
			'errors': last_error,
			'raw_results': raw_results
		}

		db.collection("creative_writing_benchmark_results").add(to_save)
		print('Creative writing benchmark results saved to firebase db.')
	except Exception as e:
		print(e)
		print('! Failed to save creative writing benchmark results to db.')


def save_judgemark_result_to_db(results, score, parseable, last_error, run_index, bench_success):
	global db

	if not db:
		return
	
	try:
		meta = results['run_metadata']

		raw_results = {}

		for i in range(meta['total_iterations']):
			iter_index = str(i+1)
			if iter_index in results['iterations']:
					raw_results[iter_index] = {
						'test_model_response': results['iterations'][iter_index]['test_model_response'],
						'judge_model_response': results['iterations'][iter_index]['judge_model_response'],
						'individual_scores': results['iterations'][iter_index]['individual_scores']
					}

		to_save = {
			'index_string': run_index,
			'run_id': meta['run_id'],
			'run_completed': int(time.time()),
			'benchmark_success': bench_success,
			'benchmark_type': 'judgemark',
			'benchmark_score': score,
			'model_path': meta['model_path'],
			'lora_path': meta['lora_path'],
			'bitsandbytes_quant': meta['bitsandbytes_quant'],
			'total_iterations': meta['total_iterations'],
			'inference_engine': meta['inference_engine'],
			'ooba_params': meta['ooba_params'],
			'include_patterns': meta['include_patterns'],
			'exclude_patterns': meta['exclude_patterns'],
			'errors': last_error,
			'raw_results': raw_results
		}

		db.collection("creative_writing_benchmark_results").add(to_save)
		print('Creative writing benchmark results saved to firebase db.')
	except Exception as e:
		print(e)
		print('! Failed to save creative writing benchmark results to db.')