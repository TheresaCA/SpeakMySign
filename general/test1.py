import requests
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Load translation model
tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)

# Hindi Translation
def translate_to_hindi(text):
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt")

    hindi_token = "hin_Deva"
    hindi_token_id = tokenizer.convert_tokens_to_ids(hindi_token)

    output_tokens = model.generate(
        **inputs,
        forced_bos_token_id=hindi_token_id,
        max_length=200
    )

    decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("🔍 Hindi Output Tokens:", decoded)
    return decoded  # ✅ THIS WAS MISSING

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
gloss = "YOU FREE TODAY"
task_type = "question"  # or "sentence"
expected_output = "Are you free today?" if task_type == "question" else "You repeat again, please."

print(f"\nGloss: {gloss}")
print(f"Task: {task_type.upper()}")

start = time.time()

# Choose prompt format
prompt = f"Convert this gloss to a {'question' if task_type == 'question' else 'proper English sentence'}: {gloss}"

# Generate English output
generated_output = run_ollama(prompt)

# Translate to Hindi
generated_output_hindi = translate_to_hindi(generated_output)

# Evaluate
bleu = compute_bleu(expected_output, generated_output)
rouge = compute_rouge(expected_output, generated_output)
duration = round(time.time() - start, 2)

# ----- Results -----
print(f"\n→ Generated Output (in English): {generated_output}")
print(f"→ Generated Output (in Hindi): {generated_output_hindi}")
print(f"→ Expected Output: {expected_output}")
print(f"→ BLEU Score: {bleu * 100:.2f}%")
print(f"→ ROUGE-L Score: {rouge * 100:.2f}%")
print(f"✓ Done in {duration}s")
