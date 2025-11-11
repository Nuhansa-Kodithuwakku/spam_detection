import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------
# Text Cleaning
# ---------------------------
def clean_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z@$% ]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------
# Numeric Features
# ---------------------------
class NumericFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array(X[['email_length', 'num_links', 'num_uppercase', 'num_special_chars']])

# ---------------------------
# Add numeric features to dataframe
# ---------------------------
def add_features(df):
    df['email_length'] = df['text'].apply(lambda x: len(str(x)))
    df['num_links'] = df['text'].apply(lambda x: len(re.findall(r"http\S+|www\S+", str(x))))
    df['num_uppercase'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    df['num_special_chars'] = df['text'].apply(lambda x: len(re.findall(r"[^a-zA-Z0-9\s]", str(x))))
    return df

