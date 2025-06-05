import requests
import time
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Load models
tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Semantic similarity model

# ---------- Translation ----------
def translate_to_kannada(text):
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt")
    kannada_token = "kan_Knda"
    kannada_token_id = tokenizer.convert_tokens_to_ids(kannada_token)
    output_tokens = model.generate(**inputs, forced_bos_token_id=kannada_token_id, max_length=200)
    decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return decoded

def translate_to_hindi(text):
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt")
    hindi_token = "hin_Deva"
    hindi_token_id = tokenizer.convert_tokens_to_ids(hindi_token)
    output_tokens = model.generate(**inputs, forced_bos_token_id=hindi_token_id, max_length=200)
    decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return decoded

def back_translate_to_english(text, target_lang):
    tokenizer.src_lang = target_lang
    inputs = tokenizer(text, return_tensors="pt")
    eng_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    output_tokens = model.generate(**inputs, forced_bos_token_id=eng_token_id, max_length=200)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# ---------- Text-to-Speech ----------
def text_to_speech(text, lang='en', filename="output.mp3"):
    tts = gTTS(text, lang=lang)
    tts.save(filename)
    print(f" Speech saved as {filename}")

def check_tts_file(filename="output.mp3"):
    return os.path.exists(filename) and os.path.getsize(filename) > 1000  # >1KB

# ---------- LLM Generation ----------
def run_ollama(prompt):
    print(f"\n--- Prompt ---\n{prompt}\n--------------")
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(" Ollama error:", str(e))
        return f"ERROR: {str(e)}"

# ---------- Semantic Similarity ----------
def semantic_similarity_score(ref, cand):
    ref_emb = embedding_model.encode(ref, convert_to_tensor=True)
    cand_emb = embedding_model.encode(cand, convert_to_tensor=True)
    return round(util.cos_sim(ref_emb, cand_emb).item(), 4)

# ---------- Gloss to Sentence/Question ----------
def handle_gloss_language(gloss, expected_output, task_type, target_language):
    # Step 1: Convert Gloss to English Sentence/Question
    if task_type == "sentence":
        prompt = f"Convert this gloss to a proper English sentence: {gloss}"
    elif task_type == "question":
        prompt = f"Convert this gloss to a proper English question: {gloss}"
    else:
        print("Invalid task type.")
        return None

    generated_output = run_ollama(prompt)
    print(f"Converted Output (in English): {generated_output}")

    # Step 2: Translate to Kannada or Hindi based on user choice
    if target_language == "kannada":
        generated_output_target = translate_to_kannada(generated_output)
        print(f"Converted Output (in Kannada): {generated_output_target}")
        text_to_speech(generated_output_target, lang='kn', filename="output_kannada.mp3")
        tts_success = check_tts_file("output_kannada.mp3")
    elif target_language == "hindi":
        generated_output_target = translate_to_hindi(generated_output)
        print(f"Converted Output (in Hindi): {generated_output_target}")
        text_to_speech(generated_output_target, lang='hi', filename="output_hindi.mp3")
        tts_success = check_tts_file("output_hindi.mp3")
    else:
        print("Unsupported target language!")
        return None

    # Step 3: Generate Speech for English
    text_to_speech(generated_output, lang='en', filename="output_english.mp3")
    tts_success_english = check_tts_file("output_english.mp3")

    # Evaluation of semantic similarity
    semantic_sim_english = semantic_similarity_score(expected_output, generated_output)
    backtranslated_english = back_translate_to_english(generated_output_target, "kan_Knda" if target_language == "kannada" else "hin_Deva")
    semantic_sim_backtranslation = semantic_similarity_score(generated_output, backtranslated_english)

    # Save the generated English sentence to a text file
    # Save the generated English and translated sentences to a text file
    with open("generated_output_hindi.txt", "w", encoding="utf-8") as f:
        f.write("Generated Output (in English):\n")
        f.write(generated_output + "\n\n")
        f.write(f"Generated Output (in {target_language.capitalize()}):\n")
        f.write(generated_output_target + "\n")


    return generated_output, generated_output_target, tts_success_english, tts_success, semantic_sim_english, backtranslated_english, semantic_sim_backtranslation

# ---------- Main Test ----------
def get_user_input():
    gloss = input("Enter the gloss in English (e.g., 'BRING WATER ME'): ").strip()
    task_type = input("Is it a sentence or a question? (type 'sentence' or 'question'): ").strip().lower()
    target_language = input("Which language do you want to translate to? (type 'kannada' or 'hindi'): ").strip().lower()
    return gloss, task_type, target_language

def main():
    gloss, task_type, target_language = get_user_input()

    expected_output = "i am sitting in the class" if task_type == "sentence" else "are you free today?"

    print(f"\nGloss: {gloss}")
    print(f"Task: {task_type.upper()}")
    print(f"Target Language: {target_language.capitalize()}")

    start = time.time()

    # Handle different languages and process the gloss
    generated_output, generated_output_target, tts_success_english, tts_success, semantic_sim_english, backtranslated_english, semantic_sim_backtranslation = handle_gloss_language(
        gloss, expected_output, task_type, target_language)

    duration = round(time.time() - start, 2)

    # ---------- Results ----------
    print(f"\n→ Generated Output (in English): {generated_output}")
    print(f"→ Generated Output (in {target_language.capitalize()}): {generated_output_target}")
    print(f"→ Expected Output: {expected_output}")
    print(f"→ Semantic Similarity (English): {semantic_sim_english * 100:.2f}%")
    print(f"→ Back-translated English: {backtranslated_english}")
    print(f"→ Semantic Similarity (Back-Translation): {semantic_sim_backtranslation * 100:.2f}%")
    print(f"→ TTS File Created: {'Yes' if all([tts_success_english, tts_success]) else 'No'}")
    print(f"✓ Done in {duration}s")

if __name__ == "__main__":
    main()
