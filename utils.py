# utils.py
import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from gtts import gTTS

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def run_ollama(prompt):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


def translate(text, lang_code):
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt")
    lang_token_id = tokenizer.convert_tokens_to_ids(lang_code)
    outputs = model.generate(**inputs, forced_bos_token_id=lang_token_id, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def text_to_speech(text, lang, filename):
    tts = gTTS(text, lang=lang)
    tts.save(filename)
    return filename if os.path.exists(filename) else None


def handle_gloss_language(gloss, task_type, target_language):
    if task_type == "sentence":
        prompt = f"Convert this gloss to a proper English sentence: {gloss}"
    elif task_type == "question":
        prompt = f"Convert this gloss to a proper English question: {gloss}"
    else:
        return "Invalid task type", None, None

    eng_sentence = run_ollama(prompt)

    if target_language == "english":
        filename = "static/output_english.mp3"
        lang_code = "en"
        translated_text = eng_sentence
    elif target_language == "hindi":
        filename = "static/output_hindi.mp3"
        translated_text = translate(eng_sentence, "hin_Deva")
        lang_code = "hi"
    elif target_language == "kannada":
        filename = "static/output_kannada.mp3"
        translated_text = translate(eng_sentence, "kan_Knda")
        lang_code = "kn"
    else:
        return "Unsupported language", None, None

    text_to_speech(translated_text, lang_code, filename)

    return translated_text, lang_code, filename
