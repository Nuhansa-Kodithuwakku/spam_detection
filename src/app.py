import os
import joblib
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates', static_folder='static')

# Absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'spam_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'model', 'vectorizer.pkl')

# Load model and vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    
    try:
        vect = vectorizer.transform(data)
        prediction = model.predict(vect)

        # FIX: reverse the label interpretation
        result = "This message is SPAM!" if prediction[0] == 0 else "This message is NOT spam."
        color = "red" if prediction[0] == 0 else "green"

        return render_template('index.html', result=result, color=color, message=message)
    except Exception as e:
        return render_template('index.html', result=f"Error: {e}", color="orange")

if __name__ == '__main__':
    app.run(debug=True)
