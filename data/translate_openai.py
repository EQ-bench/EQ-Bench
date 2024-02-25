import json
from tqdm import tqdm
import openai
import re
import sys

def translate_prompt(item, raw_file_path):
    """
    Translates a given prompt from English to German using OpenAI's translation service,
    and saves the raw response to a backup file.
    """
    # Configuration for the translation API request
    api_key = ""
    #base_url = 'http://localhost:11434/v1/'
    model = "gpt-4-0125-preview"
    temperature = 0.5
    max_tokens = 2000
    top_p = 1.0
    frequency_penalty = 0.0
    presence_penalty = 0.0
    
    try:
        item_json_string = json.dumps(item, ensure_ascii=False)
        #item_json_string = json.dumps(item, ensure_ascii=False).replace('\n', '\\n')
    except Exception as e:
        print(f"Error serializing item to JSON: {e}")
        return None
    
    # Format the prompt for translation
    translation_prompt = f"""Translate from this JSON snippet the complete English prompt to German and also all emotion names in the value fields of reference_answer and reference_answer_fullscale. Make sure you have exactly the same translation for the emotion names in the reference_answer part as in the reference_answer_fullscale part and in the two mentions in the prompt text. Also make very sure that you answer with a valid JSON snippet exactly structured as the input. Do NOT translate the keys ["prompt", "reference_answer", "reference_answer_fullscale", "emotion1", "emotion2", "emotion3", "emotion4", "emotion1_score", "emotion2_score", "emotion3_score", "emotion4_score"]. Do only answer with that JSON snippet. No explanation, comment, or anything else. Here is what we need translated:\n\n{item_json_string}"""
    
    messages = [{"role": "user", "content": translation_prompt}]
    
    try:
        # Initialize the OpenAI client with the provided API key and base URL
        #openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)
        openai_client = openai.OpenAI(api_key=api_key)
        response = openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages
        )
        
        # Extract the translated text from the response
        translated_text = response.choices[0].message.content.strip()
        #translated_text = response.choices[0].text.strip()
        translated_text = translated_text.replace("```json", "")
        translated_text = translated_text.replace("```", "")
        
        # Write the raw response to a backup file
        with open(raw_file_path, 'a', encoding='utf-8') as raw_file:
            raw_file.write(translated_text + "\n\n")
    
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return None
    
    #translated_text_escaped = translated_text.replace('\n', '\\n').replace('\_', '_')
    
    # Attempt to parse the translated JSON string back into a Python dictionary
    try:
        decoder = json.JSONDecoder(strict=False)
        translated_item = decoder.decode(translated_text)
        return translated_item
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from translated text: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
        
def validate_format(translated_data):
    """
    Validates the format of the translated data against the specified template.
    """
    required_keys = ["prompt", "reference_answer", "reference_answer_fullscale"]
    emotion_keys = ["emotion1", "emotion2", "emotion3", "emotion4"]
    score_keys = ["emotion1_score", "emotion2_score", "emotion3_score", "emotion4_score"]

    for key, item in translated_data.items():
        if not all(k in item for k in required_keys):
            return False, f"Missing one of the required keys in item {key}."
        for ref_key in ['reference_answer', 'reference_answer_fullscale']:
            ref_section = item[ref_key]
            if not all(emotion_key in ref_section for emotion_key in emotion_keys) or not all(score_key in ref_section for score_key in score_keys):
                return False, f"Missing emotion or score keys in '{ref_key}' section of item {key}."
    return True, "All items conform to the specified format."

def validate_emotion_names(item):
    """
    Validates that the output item conforms to the specified format, checking for consistency in translated emotion names.
    Specifically, it ensures each emotion name appears at least twice in the prompt and matches between reference_answer and reference_answer_fullscale.
    """
    # Extract emotion names from reference_answer
    emotion_names = [item["reference_answer"][f"emotion{i}"] for i in range(1, 5)]

    # Verify each emotion name appears at least twice in the prompt
    prompt = item["prompt"]
    for emotion_name in emotion_names:
        if prompt.count(emotion_name) < 2:
            return False, f"Emotion name does not appear at least twice."
    
    # Verify emotion names match between reference_answer and reference_answer_fullscale
    for i in range(1, 5):
        emotion_key = f"emotion{i}"
        if item["reference_answer"][emotion_key] != item["reference_answer_fullscale"][emotion_key]:
            return False, f" Emotion names do not match."

    # Passed all checks
    return True

def validate_emotion_names_v2(translated_data):
    """
    Validates that emotion names are consistent across the prompt, reference_answer, and reference_answer_fullscale.
    """
    for key, item in translated_data.items():
        emotion_names = [item["reference_answer"][f"emotion{i}"] for i in range(1, 5)]
        prompt = item["prompt"]
        for emotion_name in emotion_names:
            if prompt.count(emotion_name) < 2:
                return False, f"Emotion name '{emotion_name}' does not appear at least twice in the prompt for item {key}."
        
        for i in range(1, 5):
            emotion_key = f"emotion{i}"
            if item["reference_answer"][emotion_key] != item["reference_answer_fullscale"][emotion_key]:
                return False, f"Emotion names do not match between 'reference_answer' and 'reference_answer_fullscale' for item {key}."

    return True, "Emotion names are consistent across items."

def process_file(input_file_path, output_file_path, raw_file_path, start_index):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data_points = json.load(file)

        output_data = {}

        keys = list(data_points.keys())
        for key in tqdm(keys[start_index:], desc="Progress", unit="item"):
            item = data_points[key]
            
            # Translate the entire item
            translated_item = translate_prompt(item, raw_file_path)
            
            if translated_item is None:
                raise ValueError("Translation failed for item {}.".format(key))

            # Validate the format of the translated item
            format_valid, format_message = validate_format({key: translated_item})
            if not format_valid:
                raise ValueError(f"Format validation failed for item {key}: {format_message}")

            # Validate emotion names consistency
            emotion_names_valid, emotion_names_message = validate_emotion_names_v2({key: translated_item})
            if not emotion_names_valid:
                raise ValueError(f"Emotion names validation failed for item {key}: {emotion_names_message}")
            
            # If all validations pass, add the translated item to the output data
            output_data[key] = translated_item

            # Write the output data to a file
            with open(output_file_path, 'w', encoding='utf-8') as file:
                json.dump(output_data, file, ensure_ascii=False, indent=4)
        
        print("Translation and validation completed successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file_path = 'eq_bench_v2_questions_171_en.json'  # Update with your input file path
    output_file_path = 'eq_bench_v2_questions_171_de.json'  # Update with your output file path
    raw_file_path = 'eq_bench_v2_questions_171_de_raw.json'  # Path to the raw backup file

    start_index = 0
    if len(sys.argv) > 1:
        start_index = int(sys.argv[1])
    
    process_file(input_file_path, output_file_path, raw_file_path, start_index)
