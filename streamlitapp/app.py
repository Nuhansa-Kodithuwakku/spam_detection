import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sys

# ------------------------------------
# Project root for src imports
# ------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Dummy class definitions for unpickling
class NumericFeatures:
    def transform(self, X):
        return X
    def fit(self, X, y=None):
        return self

class TextAndFeatures:
    def transform(self, X):
        return X
    def fit(self, X, y=None):
        return self

# Utility imports
from src.utils import clean_text, add_features

# ------------------------------------
# Streamlit Page Setup with Dark Theme
# ------------------------------------
st.set_page_config(page_title="Spam/Ham Classifier", layout="centered")

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        background-color: #0E1117;
    }
    .spam-text {
        color: #FF4B4B;
        font-weight: bold;
        font-size: 1.5em;
    }
    .ham-text {
        color: #00D26A;
        font-weight: bold;
        font-size: 1.5em;
    }
    .metric-container {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .section-header {
        color: #FAFAFA;
        border-bottom: 2px solid #00D26A;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Spam / Ham Email Classifier")
st.write("Classify your emails as Spam or Ham with probability scores and explore insightful visualizations.")

# ------------------------------------
# Load model & vectorizer
# ------------------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.join(project_root, "tests/MaildataModel")
    model_path = os.path.join(base_dir, "svm_spam_model_optimized.pkl")

    model = joblib.load(model_path)

    # Try to extract vectorizer if model includes it
    vectorizer = None
    try:
        vectorizer = model.named_steps["features"].vectorizer
    except Exception:
        try:
            vectorizer = joblib.load(os.path.join(base_dir, "vectorizer.pkl"))
        except:
            st.warning("No vectorizer found inside the model.")
    return model, vectorizer


model, vectorizer = load_artifacts()

# ------------------------------------
# Helper Functions
# ------------------------------------
def prepare_input(raw_text):
    """Clean and add engineered features."""
    temp_df = pd.DataFrame({"text": [raw_text]})
    temp_df["clean_text"] = temp_df["text"].apply(clean_text)
    temp_df = add_features(temp_df)
    return temp_df


def vectorize_input(df):
    """Convert clean text to vectorized + numeric matrix."""
    if vectorizer is not None:
        tfidf_features = vectorizer.transform(df["clean_text"])
        numeric_feats = df[["email_length", "num_links", "num_uppercase", "num_special_chars"]].values
        from scipy.sparse import hstack
        X_predict = hstack([tfidf_features, numeric_feats])
    else:
        X_predict = df[
            ["clean_text", "email_length", "num_links", "num_uppercase", "num_special_chars"]
        ]
    return X_predict


def get_top_keywords(clean_email, vectorizer, top_n=5):
    if vectorizer is None:
        return ["Vectorizer unavailable"]
    tfidf_vec = vectorizer.transform([clean_email])
    feature_names = np.array(vectorizer.get_feature_names_out())
    weights = tfidf_vec.toarray().flatten()
    top_indices = weights.argsort()[-top_n:][::-1]
    top_words = [feature_names[i] for i in top_indices if weights[i] > 0]
    return top_words if top_words else ["No strong keywords detected"]


# ------------------------------------
# Session State for History
# ------------------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["Email", "Prediction", "Spam_Prob", "Ham_Prob", "Timestamp", "Input_Type"]
    )

# ------------------------------------
# Input Section
# ------------------------------------
st.markdown("## Input Email or Upload File")
input_method = st.radio("Choose input method:", ["Paste Email Text", "Upload File"])

email_text = ""
uploaded_file = None
csv_mode = False

if input_method == "Paste Email Text":
    email_text = st.text_area("Paste your email content here:", height=200)
else:
    uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith(".txt"):
            email_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if "text" in df.columns:
                csv_mode = True
            else:
                st.warning("CSV must contain a 'text' column with email content.")

# ------------------------------------
# Classification Logic
# ------------------------------------
if st.button("Classify"):
    if not email_text.strip() and not csv_mode:
        st.warning("Please enter or upload an email first.")
    else:
        # -----------------------------
        # CSV Mode
        # -----------------------------
        if csv_mode:
            results = []
            spam_count, ham_count = 0, 0
            total_emails = len(df)
            progress_bar = st.progress(0)

            df["clean_text"] = df["text"].astype(str).apply(clean_text)
            df = add_features(df)
            X_csv = vectorize_input(df)
            preds = model.predict(X_csv)
            scores = model.decision_function(X_csv)

            for idx in range(total_emails):
                score = scores[idx]
                spam_prob = 1 / (1 + np.exp(-score))
                ham_prob = 1 - spam_prob
                pred = "SPAM" if preds[idx] == 1 else "HAM"

                if preds[idx] == 1:
                    spam_count += 1
                else:
                    ham_count += 1

                results.append(
                    {
                        "Email_ID": idx + 1,
                        "Email": df["text"].iloc[idx],
                        "Prediction": pred,
                        "Spam_Prob": spam_prob * 100,
                        "Ham_Prob": ham_prob * 100,
                        "Timestamp": datetime.now(),
                    }
                )
                progress_bar.progress((idx + 1) / total_emails)

            results_df = pd.DataFrame(results)
            st.subheader("Classification Results")
            st.dataframe(
                results_df[["Email_ID", "Prediction", "Spam_Prob", "Ham_Prob"]],
                use_container_width=True,
            )

            # Dark theme pie chart
            fig, ax = plt.subplots(facecolor='#0E1117')
            ax.pie([ham_count, spam_count],
                   labels=["HAM", "SPAM"],
                   autopct="%1.1f%%",
                   colors=["#00D26A", "#FF4B4B"])
            ax.axis("equal")
            ax.set_facecolor('#0E1117')
            st.pyplot(fig)

            st.session_state.history = pd.concat(
                [st.session_state.history, results_df.assign(Input_Type="File Upload")],
                ignore_index=True,
            )

        # -----------------------------
        # Single Email Mode
        # -----------------------------
        else:
            df = prepare_input(email_text)
            X_predict = vectorize_input(df)
            score = model.decision_function(X_predict)[0]
            spam_prob = 1 / (1 + np.exp(-score))
            ham_prob = 1 - spam_prob
            pred = "SPAM" if model.predict(X_predict)[0] == 1 else "HAM"

            # Display results with colored text
            if pred == "SPAM":
                st.markdown(f'### Result: <span class="spam-text">{pred}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'### Result: <span class="ham-text">{pred}</span>', unsafe_allow_html=True)
            
            # Metrics in containers
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="metric-container">'
                           f'<h4>Spam Probability</h4>'
                           f'<h3 style="color: #FF4B4B;">{spam_prob*100:.2f}%</h3>'
                           f'</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-container">'
                           f'<h4>Ham Probability</h4>'
                           f'<h3 style="color: #00D26A;">{ham_prob*100:.2f}%</h3>'
                           f'</div>', unsafe_allow_html=True)

            # Dark theme probability bar chart
            fig, ax = plt.subplots(facecolor='#0E1117')
            ax.barh(["Ham", "Spam"], [ham_prob * 100, spam_prob * 100], color=["#00D26A", "#FF4B4B"])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", color='white')
            ax.set_ylabel("Category", color='white')
            ax.tick_params(colors='white')
            ax.set_facecolor('#0E1117')
            fig.patch.set_facecolor('#0E1117')
            st.pyplot(fig)

            top_words = get_top_keywords(df['clean_text'].iloc[0], vectorizer)
            st.markdown("#### Top Contributing Words:")
            st.write(", ".join(top_words))

            st.session_state.history = pd.concat(
                [
                    st.session_state.history,
                    pd.DataFrame(
                        {
                            "Email": [
                                email_text[:100] + ("..." if len(email_text) > 100 else "")
                            ],
                            "Prediction": [pred],
                            "Spam_Prob": [spam_prob * 100],
                            "Ham_Prob": [ham_prob * 100],
                            "Timestamp": [datetime.now()],
                            "Input_Type": ["Text Input"],
                        }
                    ),
                ],
                ignore_index=True,
            )

# ------------------------------------
# History & Insights Section
# ------------------------------------
if not st.session_state.history.empty:
    st.markdown("---")
    st.markdown('<h2 class="section-header">Classification History & Insights</h2>', unsafe_allow_html=True)

    history = st.session_state.history.copy()
    history["Timestamp"] = pd.to_datetime(history["Timestamp"])

    st.dataframe(
        history[
            ["Email", "Prediction", "Spam_Prob", "Ham_Prob", "Timestamp", "Input_Type"]
        ].tail(10),
        use_container_width=True,
    )

    st.markdown("### Overall Spam vs Ham Distribution")
    fig, ax = plt.subplots(facecolor='#0E1117')
    counts = history["Prediction"].value_counts()
    colors = ["#00D26A" if label == "HAM" else "#FF4B4B" for label in counts.index]

    # Create pie chart
    wedges, texts, autotexts = ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=90)

    # Add legend with colored boxes
    legend_labels = [f'SPAM - {counts.get("SPAM", 0)} emails', f'HAM - {counts.get("HAM", 0)} emails']
    legend_colors = ["#FF4B4B", "#00D26A"]
    ax.legend(wedges, legend_labels,
            title="Classification",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            frameon=True,
            framealpha=0.9,
            edgecolor='white')

    ax.axis("equal")
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    st.pyplot(fig)

    st.markdown("### Spam Probability Trend Over Time")
    history_sorted = history.sort_values("Timestamp")
    fig, ax = plt.subplots(facecolor='#0E1117')
    ax.plot(history_sorted["Timestamp"], history_sorted["Spam_Prob"], marker="o", color="#FF4B4B")
    ax.set_xlabel("Time", color='white')
    ax.set_ylabel("Spam Probability (%)", color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

    st.markdown("### Recent Predictions Overview")
    fig, ax = plt.subplots(facecolor='#0E1117')
    ax.bar(
        history_sorted["Timestamp"].dt.strftime("%H:%M:%S"),
        history_sorted["Spam_Prob"],
        color="#FF4B4B",
    )
    ax.set_xlabel("Timestamp", color='white')
    ax.set_ylabel("Spam Probability (%)", color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    csv = st.session_state.history.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download History as CSV",
        data=csv,
        file_name="spam_ham_history.csv",
        mime="text/csv",
    )