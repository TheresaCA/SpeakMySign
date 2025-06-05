# app.py
from flask import Flask, render_template, request, jsonify
from utils import handle_gloss_language
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Automatically extract gloss from video file name
    video_filename = "static/BRING_WATER_ME.mp4"  # You can later allow uploading to make this dynamic
    gloss = os.path.splitext(os.path.basename(video_filename))[0].replace("_", " ")
    
    language = request.form.get('language')
    task_type = request.form.get('task_type')

    result_text, lang_code, tts_path = handle_gloss_language(gloss, task_type, language)

    return jsonify({
        'sentence': result_text,
        'audio_path': '/' + tts_path
    })

if __name__ == '__main__':
    app.run(debug=True)
