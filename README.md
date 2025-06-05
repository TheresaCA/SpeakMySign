### Introduction

Communication is crucial in our day-to-day lives.
It is the basis for all human interactions, whether personal,
educational, or professional. However, the communication gap be-
tween the hearing and speech-impaired populations—especially
in a vast and densely populated country like India—remains
a major obstacle. Individuals from these communities mainly
rely on sign language to express themselves, but sign language
is not widely understood by the general public. This often
results in miscommunication, social exclusion, and limited access
to essential services. To bridge this communication divide, a
reliable translation system is needed—one that can seamlessly
interpret and translate sign language in real-time. SpeakMySign
offers a solution to this problem. It is an end-to-end translation
system that converts Indian Sign Language (ISL) into English,
Hindi, and Kannada speech using a powerful combination of
computer vision, deep learning, and advanced language models.
This innovation aims to foster greater inclusivity and accessibility,
helping the hearing and speech-impaired integrate more smoothly
into society

### Features
- ML translation of ISL to words using CNN.
- Gloss-to-sentence/question generation using LLM (Mistral via Ollama)
- Translation to Hindi or Kannada using `facebook/nllb-200-distilled-600M`
- Text-to-Speech (TTS) using Google Text-to-Speech (`gTTS`)
- Semantic similarity checking between expected and generated outputs
- Back-translation for translation accuracy verification
- MP3 file generation for all supported languages

### How It Works

User Input (via Web Form)
│
▼
[ Flask App (`app.py`) ]
│
├── Extracts gloss from video file name (e.g., BRING_WATER_ME.mp4)
│
└── Calls handle_gloss_language() in utils.py
│
▼
┌────────────────────────────────────────────┐
│ handle_gloss_language()                    │
│                                            │
│ 1. Generate English sentence using Ollama  │
│ 2. Translate sentence (if not English)     │
│ 3. Generate TTS audio using gTTS           │
└────────────────────────────────────────────┘
│
▼
Return JSON response:
- Translated Sentence
- Path to MP3 Audio File
│
▼
Display result + Play audio in browser
