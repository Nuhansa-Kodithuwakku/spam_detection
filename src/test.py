import pandas as pd
import re
import joblib
from src.utils import clean_text

# ---------------------------
# Load model + vectorizer
# ---------------------------
model = joblib.load("../tests/maildatamodel/svm_spam_model.pkl")

def predict_email_user_input():
    email_text = input("Enter the email content:\n\n")

    # Clean input
    text_clean = clean_text(email_text)

    # Extract extra features
    email_length = len(email_text)
    num_links = len(re.findall(r"http\S+|www\S+", email_text))
    num_uppercase = sum(1 for c in email_text if c.isupper())
    num_special_chars = len(re.findall(r"[^a-zA-Z0-9\s]", email_text))

    sample_df = pd.DataFrame([{
        'clean_text': text_clean,
        'email_length': email_length,
        'num_links': num_links,
        'num_uppercase': num_uppercase,
        'num_special_chars': num_special_chars
    }])

    pred = model.predict(sample_df)
    print("\nThe message is classified as:", "SPAM 🚨" if pred[0]==1 else "HAM ✅")

# predict_email_user_input()  # uncomment to test interactively
