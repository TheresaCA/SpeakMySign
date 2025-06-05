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
    print("🔍 Kannada Output Tokens:", decoded)
    return decoded

def back_translate_to_english(kannada_text):
    tokenizer.src_lang = "kan_Knda"
    inputs = tokenizer(kannada_text, return_tensors="pt")
    eng_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    output_tokens = model.generate(**inputs, forced_bos_token_id=eng_token_id, max_length=200)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# ---------- Text-to-Speech ----------
def text_to_speech_kannada(text, filename="output_kannada.mp3"):
    tts = gTTS(text, lang='kn')
    tts.save(filename)
    print(f" Kannada Speech saved as {filename}")

def check_tts_file(filename="output_kannada.mp3"):
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
        print(f" Ollama responded with status {response.status_code}")
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

# ---------- Main Test ----------
gloss = "BRING WATER ME"
task_type = "sentence"  # or "question"
expected_output = "Could you bring me some water?" if task_type == "question" else "Bring water for me."

print(f"\nGloss: {gloss}")
print(f"Task: {task_type.upper()}")

start = time.time()

prompt = f"Convert this gloss to a {'question' if task_type == 'question' else 'proper English sentence'}: {gloss}"

# Step 1: Gloss ➝ English
generated_output = run_ollama(prompt)

# Step 2: English ➝ Kannada
generated_output_kannada = translate_to_kannada(generated_output)

# Step 3: Kannada ➝ Speech
text_to_speech_kannada(generated_output_kannada)
tts_success = check_tts_file()

# Step 4: Evaluation
semantic_sim_english = semantic_similarity_score(expected_output, generated_output)
backtranslated_english = back_translate_to_english(generated_output_kannada)
semantic_sim_backtranslation = semantic_similarity_score(generated_output, backtranslated_english)

duration = round(time.time() - start, 2)

# ---------- Results ----------
print(f"\n→ Generated Output (in English): {generated_output}")
print(f"→ Generated Output (in Kannada): {generated_output_kannada}")
print(f"→ Expected Output: {expected_output}")
print(f"→ Semantic Similarity (English): {semantic_sim_english * 100:.2f}%")
print(f"→ Back-translated English: {backtranslated_english}")
print(f"→ Semantic Similarity (Back-Translation): {semantic_sim_backtranslation * 100:.2f}%")
print(f"→ TTS File Created: {'Yes' if tts_success else 'No'}")
print(f"✓ Done in {duration}s")
