import requests
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from gtts import gTTS  # Text-to-Speech for Kannada

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Load translation model
tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)

# Kannada Translation
def translate_to_kannada(text):
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt")

    kannada_token = "kan_Knda"
    kannada_token_id = tokenizer.convert_tokens_to_ids(kannada_token)

    output_tokens = model.generate(
        **inputs,
        forced_bos_token_id=kannada_token_id,
        max_length=200
    )

    decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("🔍 Kannada Output Tokens:", decoded)
    return decoded

# Generate speech in Kannada using gTTS
def text_to_speech_kannada(text, filename="output_kannada.mp3"):
    tts = gTTS(text, lang='kn')  # 'kn' is the language code for Kannada
    tts.save(filename)
    print(f" Kannada Speech saved as {filename}")

# LLM generation from Ollama
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

# BLEU score computation
def compute_bleu(reference, candidate):
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    smoothie = SmoothingFunction().method4
    return round(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie), 4)

# ROUGE-L score computation
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, candidate)
    return round(score['rougeL'].fmeasure, 4)

# ----- Test Setup -----
gloss = "BRING WATER ME"
task_type = "sentence"  # or "question"
expected_output = "Could you bring me some water?" if task_type == "question" else "Bring water for me."

print(f"\nGloss: {gloss}")
print(f"Task: {task_type.upper()}")

start = time.time()

# Choose prompt format
prompt = f"Convert this gloss to a {'question' if task_type == 'question' else 'proper English sentence'}: {gloss}"

# Generate English output
generated_output = run_ollama(prompt)

# Translate to Kannada
generated_output_kannada = translate_to_kannada(generated_output)

# Convert Kannada output to speech (save as mp3)
text_to_speech_kannada(generated_output_kannada)

# Evaluate
bleu = compute_bleu(expected_output, generated_output)
rouge = compute_rouge(expected_output, generated_output)
duration = round(time.time() - start, 2)

# ----- Results -----
print(f"\n→ Generated Output (in English): {generated_output}")
print(f"→ Generated Output (in Kannada): {generated_output_kannada}")
print(f"→ Expected Output: {expected_output}")
print(f"→ BLEU Score: {bleu * 100:.2f}%")
print(f"→ ROUGE-L Score: {rouge * 100:.2f}%")
print(f"✓ Done in {duration}s")
