import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
import joblib
import os

from src.utils import clean_text, NumericFeatures, add_features

# ---------------------------
# Custom Feature Union (module-level for pickling)
# ---------------------------
class TextAndFeatures:
    def __init__(self, vectorizer, numeric_transformer):
        self.vectorizer = vectorizer
        self.numeric_transformer = numeric_transformer
    def fit(self, X, y=None):
        self.vectorizer.fit(X['clean_text'])
        self.numeric_transformer.fit(X)
        return self
    def transform(self, X):
        text_features = self.vectorizer.transform(X['clean_text'])
        numeric_features = self.numeric_transformer.transform(X)
        return hstack([text_features, numeric_features])

# ---------------------------
# Only run training if executed directly
# ---------------------------
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../data/maildata.csv")
    df.rename(columns={'Message':'text', 'Category':'label'}, inplace=True)
    df['label_num'] = df['label'].map({'ham':0,'spam':1})

    # Preprocess + features
    df['clean_text'] = df['text'].apply(clean_text)
    df = add_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df[['clean_text', 'email_length', 'num_links', 'num_uppercase', 'num_special_chars']],
        df['label_num'],
        test_size=0.2,
        random_state=42,
        stratify=df['label_num']
    )

    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2), sublinear_tf=True)

    # Feature union
    feature_union = TextAndFeatures(tfidf, NumericFeatures())

    # Train SVM with GridSearch
    pipeline = Pipeline([
        ('features', feature_union),
        ('svm', SVC(class_weight='balanced', probability=True, random_state=42))
    ])

    param_grid = {
        'svm__C':[0.1,1,5],
        'svm__kernel':['linear','rbf'],
        'svm__gamma':['scale','auto']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("Best Params:", grid_search.best_params_)

    # Save model
    os.makedirs("../tests/MaildataModel/", exist_ok=True)
    joblib.dump(best_model, "../tests/MaildataModel/svm_spam_model_optimized.pkl")
    joblib.dump(tfidf, "../tests/MaildataModel/tfidf_vectorizer_optimized.pkl")
