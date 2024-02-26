from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import json
from tqdm import tqdm

# Initialize tokenizer and model
mname = "facebook/wmt19-en-de"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

def translate(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded
#
def translate_emotions_and_construct_prompts(input_json, output_file_path):
    translated_json = {}

    with tqdm(total=len(input_json), desc="Translating", unit="item") as pbar:
        for key, value in input_json.items():
            # Extract and translate emotion names from reference_answer
            emotion_names = {k: v for k, v in value['reference_answer'].items() if not k.endswith('_score')}
            translated_emotions = {k: translate(v) for k, v in emotion_names.items()}

            # Split the prompt into segments, translate each, and reassemble with \n preserved
            prompt_segments = value['prompt'].split("\n")
            translated_segments = [translate(segment) for segment in prompt_segments if segment.strip()]
            full_translated_prompt = "\n".join(translated_segments)

            # Construct new prompt using translated emotion names
            # Assuming the last part of the prompt needs the new template with emotions
            emotions_formatted = [translated_emotions[f"emotion{i+1}"] for i in range(4)]
            new_prompt_template = "Ergebnisse des ersten Durchgangs:\n{}: <score>\n{}: <score>\n{}: <score>\n{}: <score>\n\nKritik: <Ihre Kritik hier>\n\nÜberarbeitete Ergebnisse:\n{}: <revidierte Bewertung>\n{}: <revidierte Bewertung>\n{}: <überarbeitete Wertung>\n{}: <überarbeitete Wertung>\n\n[Ende der Antwort]\n\nBeachte: Null ist ein gültiger Wert, das heißt, sie empfinden diese Emotion wahrscheinlich nicht. Sie müssen mindestens eine Emotion > 0 erreichen."
            full_translated_prompt += "\n\n" + new_prompt_template.format(*emotions_formatted, *emotions_formatted)

            translated_json[key] = {
                "prompt": full_translated_prompt,
                "reference_answer": {**translated_emotions, **{k: v for k, v in value['reference_answer'].items() if k.endswith('_score')}},
                "reference_answer_fullscale": {**translated_emotions, **{k: v for k, v in value['reference_answer_fullscale'].items() if k.endswith('_score')}}
            }

            # Print the translated section for verbose checking
            #print(f"Translated section {key}:")
            #print(json.dumps(translated_json[key], ensure_ascii=False, indent=4))
            #print("\n" + "="*50 + "\n")
            
            pbar.update(1)

    # Write the entire translated JSON to a file at the end
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(translated_json, file, ensure_ascii=False, indent=4)

input_file_path = 'eq_bench_v2_questions_171_en.json'  # Update with your input file path
output_file_path = 'eq_bench_v2_questions_171_de_wmt.json'  # Update with your output file path

# Load input JSON
with open(input_file_path, 'r', encoding='utf-8') as file:
    json_input = json.load(file)

# Translate emotions and prompts, and print each section as it's processed
translate_emotions_and_construct_prompts(json_input, output_file_path)        

