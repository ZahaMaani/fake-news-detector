import streamlit as st
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Fake News Detector", page_icon="🕵️")

@st.cache_resource
def setup_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

setup_nltk()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_news_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    important_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(important_words)

@st.cache_resource
def load_assets():
    model_path = '../models/fake_news_model.pkl'
    vectorizer_path = '../models/tfidf_vectorizer.pkl'
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except FileNotFoundError:
    st.error("Model files not found in /models. Please run 'src/train.py' first.")
    st.stop()

st.title("🕵️ Fake News Detection System")
st.write("Analyze news content authenticity using Machine Learning.")

user_input = st.text_area("Paste news article text here:", height=300)

if st.button("Analyze Content"):
    if user_input.strip():
        cleaned = clean_news_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        if prediction == "Fake":
            st.error(f"### Prediction: {prediction}")
            st.warning("Warning: This content matches patterns often found in fake news.")
        else:
            st.success(f"### Prediction: {prediction}")
            st.info("Information: This content appears consistent with real news reporting.")
    else:
        st.warning("Please enter text to begin analysis.")