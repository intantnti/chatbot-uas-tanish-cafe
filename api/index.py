from flask import Flask, request, jsonify, render_template
import pickle
import os
import re
import random

app = Flask(__name__, template_folder='../templates')

# Load Model Absolute Path (Penting untuk Vercel)
base_path = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_path, '../model')

# Load semua file pickle
with open(os.path.join(model_dir, 'chat_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)
with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)
with open(os.path.join(model_dir, 'responses.pkl'), 'rb') as f:
    responses = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({'response': "Pesan kosong."})

    # Preprocessing
    clean_input = clean_text(user_input)
    
    # Feature Extraction
    input_vec = vectorizer.transform([clean_input])
    
    # Prediksi
    # Cek probabilitas tertinggi
    probs = model.predict_proba(input_vec)[0]
    max_prob = max(probs)
    pred_index = model.predict(input_vec)[0]
    pred_tag = le.inverse_transform([pred_index])[0]
    
    # Confidence Threshold (Jika AI bingung / < 60% yakin)
    if max_prob < 0.45: 
        response_text = random.choice(responses['fallback']) # Ambil respon fallback yang sudah kita buat santai
        intent = "unknown"
    else:
        response_text = random.choice(responses[pred_tag])
        intent = pred_tag

    return jsonify({
        'response': response_text,
        'intent': intent,
        'confidence': float(max_prob)
    })

# Handler untuk Vercel Serverless
if __name__ == '__main__':
    app.run(debug=True)