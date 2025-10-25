import os
import re
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from datetime import datetime

# Optional Gmail imports
try:
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    import pickle
except ImportError:
    pass  # Gmail feature requires google-api-python-client & google-auth-oauthlib



# Streamlit Page Setup
#st.set_page_config configure the web page layout and title
#st.write add descriptive texts
st.set_page_config(page_title="Spam/Ham Classifier", layout="centered")
st.title("Spam / Ham Email Classifier")
st.write(
    "Check whether your email is **Spam** or **Ham**, see confidence scores, "
    "and explore keyword importance."
)


# Load model & vectorizer
#@st.cache_resource: Caches the loaded model and vectorizer to avoid reloading on every interaction.
#joblib.load used to loads the saved svm model and tf-idf vectorizer
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "svm_spam_model.pkl")
    vect_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)
    return model, vectorizer

model, vectorizer = load_artifacts()


# Text cleaning
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'[^a-z0-9@$% ]', ' ', text)
    words = text.split()
    return ' '.join(words)


# Get top keywords
def get_top_keywords(text, vectorizer, top_n=5):
    """Return top contributing keywords based on TF-IDF weights."""
    clean = clean_text(text)
    tfidf_vec = vectorizer.transform([clean])
    feature_names = np.array(vectorizer.get_feature_names_out())
    weights = tfidf_vec.toarray().flatten()
    top_indices = weights.argsort()[-top_n:][::-1]
    return feature_names[top_indices]


# Session State
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Email", "Prediction", "Confidence", "Timestamp", "Input_Type"])

if "gmail_connected" not in st.session_state:
    st.session_state.gmail_connected = False




# Email Input or File Upload
st.markdown("## ✉️ Paste Email or Upload File")

input_method = st.radio("Choose input method:", 
                       ["Paste Email Text", "Upload File"])

email_text = ""
uploaded_file = None
csv_mode = False

if input_method == "Paste Email Text":
    email_text = st.text_area("Paste your email content here:", height=200, placeholder="Type or paste your email...")
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


# Classification Logic
if st.button("🚀 Classify"):
    if not email_text.strip() and not csv_mode:
        st.warning("Please enter or upload an email first.")
    else:
        if csv_mode:
            # --- Handle CSV upload - Show table with emails, predictions, and confidence scores ---
            results = []
            spam_count = 0
            ham_count = 0
            confidence_scores = []
            
            st.subheader("📋 File Classification Results")
            
            # Create progress bar
            progress_bar = st.progress(0)
            total_emails = len(df)
            
            for idx, row in enumerate(df.iterrows()):
                _, data = row
                text = str(data["text"])
                clean_email = clean_text(text)
                X = vectorizer.transform([clean_email])
                pred = model.predict(X)[0]
                confidence = model.decision_function(X)[0]
                confidence_score = 1 / (1 + np.exp(-confidence))
                confidence_percent = confidence_score * 100
                confidence_scores.append(confidence_percent)
                
                # Count spam/ham for visualization
                if pred == 1:
                    spam_count += 1
                    label = "SPAM 🚨"
                    badge_color = "red"
                else:
                    ham_count += 1
                    label = "HAM ✅"
                    badge_color = "green"
                
                results.append({
                    "Email_ID": idx + 1,
                    "Email_Content": text,
                    "Prediction": label,
                    "Confidence_Score": f"{confidence_percent:.1f}%",
                    "Numeric_Confidence": confidence_percent,
                    "Badge_Color": badge_color,
                    "Timestamp": datetime.now()
                })
                
                # Update progress bar
                progress_bar.progress((idx + 1) / total_emails)
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Display individual emails in columns
            st.subheader("📧 Individual Email Analysis")
            
            # Display emails in a grid layout
            cols_per_row = 2
            total_emails = len(results_df)
            
            for i in range(0, total_emails, cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < total_emails:
                        email_data = results_df.iloc[i + j]
                        with cols[j]:
                            # Create a card-like container
                            with st.container():
                                st.markdown(f"""
                                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; 
                                            border-left: 5px solid {email_data['Badge_Color']};">
                                    <h4 style="margin: 0; color: {email_data['Badge_Color']};">{email_data['Prediction']}</h4>
                                    <p style="color: #666; font-size: 0.9em; margin: 5px 0;">Confidence: {email_data['Confidence_Score']}</p>
                                    <div style="max-height: 150px; overflow-y: auto; border: 1px solid #eee; padding: 8px; 
                                                border-radius: 5px; background-color: #f9f9f9;">
                                        {email_data['Email_Content'][:300]}{'...' if len(email_data['Email_Content']) > 300 else ''}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
            
            # Display the table view as well
            st.subheader("📊 Tabular View")
            display_df = results_df[["Email_ID", "Prediction", "Confidence_Score"]].copy()
            display_df["Email_Preview"] = results_df["Email_Content"].apply(
                lambda x: x[:100] + ("..." if len(x) > 100 else "")
            )
            st.dataframe(display_df[["Email_ID", "Email_Preview", "Prediction", "Confidence_Score"]], 
                        use_container_width=True)
            
            # Display Overall Distribution and Performance Metrics
            st.subheader("📈 File Analysis Summary")
            
            # Create two columns for charts and metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Overall Distribution Pie Chart
                st.markdown("#### 📊 Overall Distribution")
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                labels = ['HAM ✅', 'SPAM 🚨']
                sizes = [ham_count, spam_count]
                colors = ['#99FF99', '#FF9999']
                
                if total_emails > 0:
                    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                           startangle=90, shadow=True)
                    ax1.axis('equal')
                    ax1.set_title('Spam vs Ham Distribution')
                else:
                    ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('Spam vs Ham Distribution')
                
                st.pyplot(fig1)
            
            with col2:
                # Performance Metrics
                st.markdown("#### 📈 Performance Metrics")
                
                if total_emails > 0:
                    spam_percentage = (spam_count / total_emails) * 100
                    ham_percentage = (ham_count / total_emails) * 100
                    avg_confidence = np.mean(confidence_scores)
                    
                    st.metric("Total Emails Analyzed", total_emails)
                    st.metric("Spam Emails", f"{spam_count} ({spam_percentage:.1f}%)")
                    st.metric("Ham Emails", f"{ham_count} ({ham_percentage:.1f}%)")
                    st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                    
                    # Additional metrics
                    high_confidence = len([score for score in confidence_scores if score >= 80])
                    st.metric("High Confidence (≥80%)", f"{high_confidence} ({(high_confidence/total_emails)*100:.1f}%)")
                else:
                    st.info("No data available for metrics")
            
            # Save to history with file type
            file_history_data = results_df[["Email_Content", "Prediction", "Confidence_Score", "Timestamp"]].rename(
                columns={"Email_Content": "Email", "Confidence_Score": "Confidence"})
            file_history_data["Input_Type"] = "File Upload"
            
            st.session_state.history = pd.concat([st.session_state.history, file_history_data], ignore_index=True)

        elif uploaded_file and not csv_mode:
            # --- Handle single file upload (TXT) - Show confidence score only ---
            clean_email = clean_text(email_text)
            X = vectorizer.transform([clean_email])
            pred = model.predict(X)[0]
            confidence = model.decision_function(X)[0]
            confidence_score = 1 / (1 + np.exp(-confidence))
            
            st.markdown("### 📊 File Classification Result")
            st.markdown(f"**Confidence Score:** {confidence_score*100:.2f}%")
            
            # Display the email content in a expandable section
            with st.expander("📧 View Email Content"):
                st.text_area("Email Content", email_text, height=200, key="file_email_content")
            
            # Don't show spam/ham label for file uploads
            st.info("For file uploads, only confidence scores are shown. For spam/ham classification, paste email text directly.")

            # Add to history with file type
            st.session_state.history = pd.concat([
                st.session_state.history,
                pd.DataFrame({
                    "Email": [email_text[:100] + ("..." if len(email_text) > 100 else "")],
                    "Prediction": [f"Score: {confidence_score*100:.1f}%"],
                    "Confidence": [f"{confidence_score*100:.1f}%"],
                    "Timestamp": [datetime.now()],
                    "Input_Type": ["File Upload"]
                })
            ], ignore_index=True)

        else:
            # --- Handle single pasted email - Show spam/ham classification ---
            clean_email = clean_text(email_text)
            X = vectorizer.transform([clean_email])
            pred = model.predict(X)[0]
            confidence = model.decision_function(X)[0]
            confidence_score = 1 / (1 + np.exp(-confidence))
            label = "SPAM 🚨" if pred == 1 else "HAM ✅"
            color = "red" if pred == 1 else "green"

            # Display in a nice card format
            st.markdown(f"""
            <div style="border: 2px solid {color}; border-radius: 10px; padding: 20px; margin: 20px 0;
                        background-color: {color}11; border-left: 10px solid {color};">
                <h2 style="margin: 0; color: {color};">Result: {label}</h2>
                <p style="font-size: 1.2em; margin: 10px 0;"><strong>Confidence:</strong> {confidence_score*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            top_words = get_top_keywords(email_text, vectorizer)
            st.markdown("#### 🔍 Top Contributing Words:")
            st.write(", ".join([f"**{w}**" for w in top_words]))

            # Display the analyzed email
            with st.expander("📧 View Analyzed Email Content"):
                st.text_area("Email Content", email_text, height=200, key="pasted_email_content")

            # Add to history with text type
            st.session_state.history = pd.concat([
                st.session_state.history,
                pd.DataFrame({
                    "Email": [email_text[:100] + ("..." if len(email_text) > 100 else "")],
                    "Prediction": [label],
                    "Confidence": [f"{confidence_score*100:.1f}%"],
                    "Timestamp": [datetime.now()],
                    "Input_Type": ["Text Input"]
                })
            ], ignore_index=True)


# ----------------------------
# Gmail Integration (Optional)
# ----------------------------
# st.markdown("---")
# st.markdown("## 🔗 Connect Your Gmail (Optional)")

# GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# def gmail_authenticate():
#     if os.path.exists("token.pickle"):
#         with open("token.pickle", "rb") as token:
#             creds = pickle.load(token)
#     else:
#         if not os.path.exists("credentials.json"):
#             st.warning("⚠️ Place your 'credentials.json' for Gmail API in the project folder.")
#             return None
#         flow = Flow.from_client_secrets_file(
#             "credentials.json",
#             scopes=GMAIL_SCOPES,
#             redirect_uri="http://localhost:8501/"
#         )
#         auth_url, _ = flow.authorization_url(prompt="consent")
#         st.markdown(f"[Click here to authorize Gmail]({auth_url})")
#         st.stop()
#     service = build('gmail', 'v1', credentials=creds)
#     return service

# if st.button("Connect Gmail"):
#     service = gmail_authenticate()
#     if service:
#         st.session_state.gmail_connected = True
#         st.success("✅ Gmail connected successfully!")


# ----------------------------
# Gmail Email Classification
# ----------------------------
# if st.session_state.gmail_connected:
#     st.markdown("### 📬 Fetch & Classify Gmail Emails")
#     if st.button("Fetch Emails"):
#         service = gmail_authenticate()
#         results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=10).execute()
#         messages = results.get('messages', [])
#         gmail_emails = []

#         for msg in messages:
#             msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
#             payload = msg_data.get('payload', {})
#             parts = payload.get('parts', [])
#             for part in parts:
#                 if part.get('mimeType') == 'text/plain':
#                     body = part['body'].get('data')
#                     if body:
#                         text = base64.urlsafe_b64decode(body).decode()
#                         gmail_emails.append(text)

#         classified = []
#         spam_count = 0
#         ham_count = 0
#         confidence_scores = []
        
#         for i, e in enumerate(gmail_emails):
#             X = vectorizer.transform([clean_text(e)])
#             pred = model.predict(X)[0]
#             confidence = model.decision_function(X)[0]
#             confidence_score = 1 / (1 + np.exp(-confidence))
#             confidence_percent = confidence_score * 100
#             confidence_scores.append(confidence_percent)
#             label = "SPAM 🚨" if pred == 1 else "HAM ✅"
            
#             if pred == 1:
#                 spam_count += 1
#                 badge_color = "red"
#             else:
#                 ham_count += 1
#                 badge_color = "green"
                
#             classified.append({
#                 "Email_ID": i + 1,
#                 "Email_Content": e,
#                 "Prediction": label,
#                 "Confidence": f"{confidence_percent:.1f}%",
#                 "Badge_Color": badge_color,
#                 "Timestamp": datetime.now()
#             })

#         # Display Gmail emails in column-wise format
#         st.subheader("📧 Gmail Email Analysis")
        
#         cols_per_row = 2
#         total_emails = len(classified)
        
#         for i in range(0, total_emails, cols_per_row):
#             cols = st.columns(cols_per_row)
#             for j in range(cols_per_row):
#                 if i + j < total_emails:
#                     email_data = classified[i + j]
#                     with cols[j]:
#                         st.markdown(f"""
#                         <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; 
#                                     border-left: 5px solid {email_data['Badge_Color']};">
#                             <h4 style="margin: 0; color: {email_data['Badge_Color']};">{email_data['Prediction']}</h4>
#                             <p style="color: #666; font-size: 0.9em; margin: 5px 0;">Confidence: {email_data['Confidence']}</p>
#                             <div style="max-height: 150px; overflow-y: auto; border: 1px solid #eee; padding: 8px; 
#                                         border-radius: 5px; background-color: #f9f9f9;">
#                                 {email_data['Email_Content'][:300]}{'...' if len(email_data['Email_Content']) > 300 else ''}
#                             </div>
#                         </div>
#                         """, unsafe_allow_html=True)

#         # Display Overall Distribution and Performance Metrics for Gmail
#         st.subheader("📈 Gmail Analysis Summary")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Overall Distribution Pie Chart
#             st.markdown("#### 📊 Overall Distribution")
#             fig1, ax1 = plt.subplots(figsize=(6, 6))
#             labels = ['HAM ✅', 'SPAM 🚨']
#             sizes = [ham_count, spam_count]
#             colors = ['#99FF99', '#FF9999']
            
#             if total_emails > 0:
#                 ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
#                        startangle=90, shadow=True)
#                 ax1.axis('equal')
#                 ax1.set_title('Gmail Spam/Ham Distribution')
#             else:
#                 ax1.text(0.5, 0.5, 'No emails found', ha='center', va='center', transform=ax1.transAxes)
#                 ax1.set_title('Gmail Spam/Ham Distribution')
            
#             st.pyplot(fig1)
        
#         with col2:
#             # Performance Metrics
#             st.markdown("#### 📈 Performance Metrics")
            
#             if total_emails > 0:
#                 spam_percentage = (spam_count / total_emails) * 100
#                 ham_percentage = (ham_count / total_emails) * 100
#                 avg_confidence = np.mean(confidence_scores)
                
#                 st.metric("Total Emails", total_emails)
#                 st.metric("Spam Emails", f"{spam_count} ({spam_percentage:.1f}%)")
#                 st.metric("Ham Emails", f"{ham_count} ({ham_percentage:.1f}%)")
#                 st.metric("Average Confidence", f"{avg_confidence:.1f}%")
#             else:
#                 st.info("No emails available for analysis")

#         new_hist = pd.DataFrame({
#             "Email": [t["Email_Content"][:100] + "..." for t in classified],
#             "Prediction": [t["Prediction"] for t in classified],
#             "Confidence": [t["Confidence"] for t in classified],
#             "Timestamp": [t["Timestamp"] for t in classified],
#             "Input_Type": ["Gmail"] * len(classified)
#         })
#         st.session_state.history = pd.concat([st.session_state.history, new_hist], ignore_index=True)


# ----------------------------
# History & Dashboard
# ----------------------------
if not st.session_state.history.empty:
    st.markdown("---")
    st.subheader("📊 Classification History & Analytics")
    
    # Display the complete history table
    st.dataframe(st.session_state.history.tail(10))
    
    # Create separate historical summaries for different input types
    history_df = st.session_state.history.copy()
    
    # Filter data by input type
    text_history = history_df[history_df["Input_Type"] == "Text Input"]
    file_history = history_df[history_df["Input_Type"] == "File Upload"]
    gmail_history = history_df[history_df["Input_Type"] == "Gmail"]
    
    # Create tabs for different historical summaries
    tab1, tab2, tab3 = st.tabs(["📋 All History", "📝 Text Input History", "📁 File Upload History"])
    
    with tab1:
        # Overall Historical Summary
        st.subheader("📈 Overall Historical Summary")
        
        if not history_df.empty:
            # Filter only spam/ham classifications for charts
            spam_ham_history = history_df[history_df["Prediction"].str.contains("SPAM|HAM", na=False)]
            
            if not spam_ham_history.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Overall Distribution Pie Chart
                    st.markdown("#### 📊 Overall Distribution")
                    fig1, ax1 = plt.subplots(figsize=(6, 6))
                    counts = spam_ham_history["Prediction"].value_counts()
                    colors = ['#99FF99', '#FF9999']
                    
                    ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, 
                           colors=colors, shadow=True)
                    ax1.axis("equal")
                    ax1.set_title('Overall Spam/Ham Distribution')
                    st.pyplot(fig1)
                
                with col2:
                    # Performance Metrics
                    st.markdown("#### 📈 Performance Metrics")
                    
                    total_classifications = len(spam_ham_history)
                    spam_count = len(spam_ham_history[spam_ham_history['Prediction'] == 'SPAM 🚨'])
                    ham_count = len(spam_ham_history[spam_ham_history['Prediction'] == 'HAM ✅'])
                    
                    # Input type distribution
                    input_counts = history_df['Input_Type'].value_counts()
                    
                    st.metric("Total Classifications", total_classifications)
                    st.metric("Spam Rate", f"{(spam_count/total_classifications)*100:.1f}%")
                    st.metric("Ham Rate", f"{(ham_count/total_classifications)*100:.1f}%")
                    
                    # Input type summary
                    st.metric("Text Inputs", len(text_history))
                    st.metric("File Uploads", len(file_history))
                    if not gmail_history.empty:
                        st.metric("Gmail Emails", len(gmail_history))
            else:
                st.info("No classification data available in history")
        else:
            st.info("No historical data available")
    
    with tab2:
        # Text Input Historical Summary
        st.subheader("📝 Text Input Historical Summary")
        
        if not text_history.empty:
            # Filter only spam/ham classifications
            text_spam_ham = text_history[text_history["Prediction"].str.contains("SPAM|HAM", na=False)]
            
            if not text_spam_ham.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Text Input Distribution
                    st.markdown("#### 📊 Text Input Distribution")
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    counts = text_spam_ham["Prediction"].value_counts()
                    colors = ['#99FF99', '#FF9999']
                    
                    ax2.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, 
                           colors=colors, shadow=True)
                    ax2.axis("equal")
                    ax2.set_title('Text Input Spam/Ham Distribution')
                    st.pyplot(fig2)
                
                with col2:
                    # Text Input Metrics
                    st.markdown("#### 📈 Text Input Metrics")
                    
                    total_text = len(text_spam_ham)
                    text_spam_count = len(text_spam_ham[text_spam_ham['Prediction'] == 'SPAM 🚨'])
                    text_ham_count = len(text_spam_ham[text_spam_ham['Prediction'] == 'HAM ✅'])
                    
                    st.metric("Total Text Classifications", total_text)
                    st.metric("Text Spam Rate", f"{(text_spam_count/total_text)*100:.1f}%")
                    st.metric("Text Ham Rate", f"{(text_ham_count/total_text)*100:.1f}%")
                    st.metric("Unique Text Emails", text_history['Email'].nunique())
                    
                    # Recent text classifications
                    st.markdown("#### 📅 Recent Text Classifications")
                    recent_text = text_history.tail(5)
                    for _, row in recent_text.iterrows():
                        pred_color = "red" if "SPAM" in row['Prediction'] else "green"
                        st.markdown(f"**{row['Email']}** - <span style='color:{pred_color}'>{row['Prediction']}</span>", 
                                  unsafe_allow_html=True)
            else:
                st.info("No text classification data available")
        else:
            st.info("No text input history available")
    
    with tab3:
        # File Upload Historical Summary
        st.subheader("📁 File Upload Historical Summary")
        
        if not file_history.empty:
            # For file uploads, we need to handle both spam/ham and confidence scores
            file_spam_ham = file_history[file_history["Prediction"].str.contains("SPAM|HAM", na=False)]
            file_scores = file_history[file_history["Prediction"].str.contains("Score:", na=False)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not file_spam_ham.empty:
                    # File Upload Distribution
                    st.markdown("#### 📊 File Upload Distribution")
                    fig3, ax3 = plt.subplots(figsize=(6, 6))
                    counts = file_spam_ham["Prediction"].value_counts()
                    colors = ['#99FF99', '#FF9999']
                    
                    ax3.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, 
                           colors=colors, shadow=True)
                    ax3.axis("equal")
                    ax3.set_title('File Upload Spam/Ham Distribution')
                    st.pyplot(fig3)
                else:
                    st.markdown("#### 📊 File Upload Summary")
                    st.info("No spam/ham classifications in file uploads")
            
            with col2:
                # File Upload Metrics
                st.markdown("#### 📈 File Upload Metrics")
                
                total_files = len(file_history)
                file_spam_count = len(file_spam_ham[file_spam_ham['Prediction'] == 'SPAM 🚨'])
                file_ham_count = len(file_spam_ham[file_spam_ham['Prediction'] == 'HAM ✅'])
                score_count = len(file_scores)
                
                st.metric("Total File Uploads", total_files)
                if file_spam_count + file_ham_count > 0:
                    st.metric("Files with Spam/Ham", file_spam_count + file_ham_count)
                    st.metric("File Spam Rate", f"{(file_spam_count/(file_spam_count + file_ham_count))*100:.1f}%")
                st.metric("Files with Scores Only", score_count)
                st.metric("Unique File Uploads", file_history['Email'].nunique())
                
                # File upload statistics
                if total_files > 0:
                    st.markdown("#### 📊 File Upload Types")
                    st.metric("CSV Files", len(file_spam_ham))
                    st.metric("TXT Files", score_count)
        else:
            st.info("No file upload history available")
    
    # with tab4:
    #     # Gmail Historical Summary
    #     st.subheader("📧 Gmail Historical Summary")
        
    #     if not gmail_history.empty:
    #         gmail_spam_ham = gmail_history[gmail_history["Prediction"].str.contains("SPAM|HAM", na=False)]
            
    #         if not gmail_spam_ham.empty:
    #             col1, col2 = st.columns(2)
                
    #             with col1:
    #                 # Gmail Distribution
    #                 st.markdown("#### 📊 Gmail Distribution")
    #                 fig4, ax4 = plt.subplots(figsize=(6, 6))
    #                 counts = gmail_spam_ham["Prediction"].value_counts()
    #                 colors = ['#99FF99', '#FF9999']
                    
    #                 ax4.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, 
    #                        colors=colors, shadow=True)
    #                 ax4.axis("equal")
    #                 ax4.set_title('Gmail Spam/Ham Distribution')
    #                 st.pyplot(fig4)
                
    #             with col2:
    #                 # Gmail Metrics
    #                 st.markdown("#### 📈 Gmail Metrics")
                    
    #                 total_gmail = len(gmail_spam_ham)
    #                 gmail_spam_count = len(gmail_spam_ham[gmail_spam_ham['Prediction'] == 'SPAM 🚨'])
    #                 gmail_ham_count = len(gmail_spam_ham[gmail_spam_ham['Prediction'] == 'HAM ✅'])
                    
    #                 st.metric("Total Gmail Classifications", total_gmail)
    #                 st.metric("Gmail Spam Rate", f"{(gmail_spam_count/total_gmail)*100:.1f}%")
    #                 st.metric("Gmail Ham Rate", f"{(gmail_ham_count/total_gmail)*100:.1f}%")
    #                 st.metric("Unique Gmail Sessions", gmail_history['Timestamp'].dt.date.nunique())
                    
    #                 # Recent Gmail activity
    #                 if total_gmail > 0:
    #                     last_session = gmail_history['Timestamp'].max()
    #                     st.metric("Last Gmail Fetch", last_session.strftime("%Y-%m-%d %H:%M"))
    #         else:
    #             st.info("No Gmail classification data available")
    #     else:
    #         st.info("No Gmail history available")
    
    # Download button for complete history
    csv = st.session_state.history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Complete History as CSV",
        data=csv,
        file_name="spam_ham_complete_history.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("💡 Built by Amie — Spam Detection App with explainable AI, Gmail integration, and export features.")