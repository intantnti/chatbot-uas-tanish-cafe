import json
import pickle
import re
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# 1. Load Dataset (UTF-8 Fix)
print("[1/4] Loading Dataset...")
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

patterns = []
tags = []
responses = {}

# 2. Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

for intent in intents['intents']:
    responses[intent['tag']] = intent['responses']
    if intent['tag'] != 'fallback':
        for pattern in intent['patterns']:
            cleaned_pattern = clean_text(pattern)
            patterns.append(cleaned_pattern)
            tags.append(intent['tag'])

# 3. Feature Extraction
print("[2/4] Feature Extraction...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

le = LabelEncoder()
y = le.fit_transform(tags)

# 4. Training Model
# Kita gunakan 100% data untuk training agar bot maksimal
print("[3/4] Training Neural Network (LBFGS)...")

# Menggunakan solver 'lbfgs' (Paling bagus untuk data kecil < 1000 baris)
model = MLPClassifier(
    hidden_layer_sizes=(16, 16), 
    activation='relu',
    solver='lbfgs',      # Algoritma khusus dataset kecil
    max_iter=2000,       # Durasi belajar lebih lama
    random_state=42
)

# INI BARIS YANG TADI HILANG:
model.fit(X, y) 

print(f"--> Training Selesai! Model mempelajari {len(patterns)} pola kalimat.")

# 5. Simpan Model
print("[4/4] Menyimpan Model...")
with open('model/chat_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('model/responses.pkl', 'wb') as f:
    pickle.dump(responses, f)

print("\nSUKSES! Silakan jalankan 'python api/index.py' sekarang.")