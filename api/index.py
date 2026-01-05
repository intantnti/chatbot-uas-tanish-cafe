from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import random

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# --- BAGIAN PENTING: MENCARI LOKASI FILE DENGAN PASTI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

model_path = os.path.join(base_dir, 'model', 'chat_model.pkl')
vectorizer_path = os.path.join(base_dir, 'model', 'vectorizer.pkl')
le_path = os.path.join(base_dir, 'model', 'label_encoder.pkl')
responses_path = os.path.join(base_dir, 'model', 'responses.pkl')

if not os.path.exists(model_path):
    print(f"ERROR FATAL: File model tidak ditemukan di {model_path}")

# Load Model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
    with open(responses_path, 'rb') as f:
        responses = pickle.load(f)
except Exception as e:
    print(f"Error Loading Model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message')

        if not user_input:
            return jsonify({'response': "Ketik sesuatu dong kak :)"})

        # Preprocessing & Prediksi
        input_vec = vectorizer.transform([user_input.lower()])
        prediction = model.predict(input_vec)[0]
        probs = model.predict_proba(input_vec)[0]
        max_prob = np.max(probs)

        # Threshold Confidence
        if max_prob < 0.5:
            response_text = random.choice(responses['fallback'])
        else:
            predicted_tag = le.inverse_transform([prediction])[0]
            response_text = random.choice(responses[predicted_tag])

        return jsonify({'response': response_text})

    except Exception as e:
        return jsonify({'response': f"Maaf, ada error sistem: {str(e)}"})

# --- BAGIAN INI HARUS DI PALING KIRI (JANGAN DI TAB) ---
if __name__ == '__main__':
    print("Mulai menjalankan server lokal...")
    # Baris di bawah ini BARU di-tab (masuk ke dalam if)
    app.run(port=5000, debug=True)