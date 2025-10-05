import streamlit as st
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 


try:
    
    model = joblib.load('fake_news_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    st.sidebar.success("Model and Vectorizer loaded successfully.")
except Exception:
    st.sidebar.error("ERROR: Model files not found. Please run 'python training_code.py' first.")
    st.stop()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    return text.strip()


def predict_news(text):
    cleaned_text = clean_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    
    
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0] 
    
    
    confidence = probabilities[prediction]
        
    return prediction, confidence


st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown('<style>h1 {color: #1E40AF;} .stButton>button {background-color: #3B82F6; color: white; border-radius: 8px;}</style>', unsafe_allow_html=True)
st.title("📰 ML Fake News Detector")
st.markdown("Paste an article below to check if it's **Real** or **Fake** using your trained model.")
st.markdown("---")

input_text = st.text_area("Paste News Article Text Here:", 
                          height=300, 
                          placeholder="Type or paste a news article...")

if st.button("Classify Article", use_container_width=True):
    if input_text:
        prediction, confidence = predict_news(input_text)
        confidence_percent = confidence * 100
        
        st.markdown("## 🔍 Classification Result")
        
        if prediction == 0:
            st.error("🚨 FAKE NEWS ALERT 🚨")
            st.markdown(f"<h3 style='color: red;'>Result: Likely FAKE NEWS</h3>", unsafe_allow_html=True)
            st.info(f"Confidence: **{confidence_percent:.2f}%**")
        else:
            st.success("✅ REAL NEWS")
            st.markdown(f"<h3 style='color: green;'>Result: Likely REAL NEWS</h3>", unsafe_allow_html=True)
            st.info(f"Confidence: **{confidence_percent:.2f}%**")
            
    else:
        st.warning("Please paste some text into the box to classify.")

st.markdown("---")
st.markdown(f"<sub>Model: Logistic Regression | F1-Score: approx 97.6% | Deployment: Streamlit</sub>")
