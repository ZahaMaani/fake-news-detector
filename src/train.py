import pandas as pd
import re
import joblib
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_news_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    important_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(important_words)

# Load data
try:
    df_fake = pd.read_csv('../data/Fake.csv')
    df_true = pd.read_csv('../data/True.csv')
except FileNotFoundError:
    print("Error: Ensure CSV files are in the 'data' folder at the root.")
    exit()

# Preprocessing logic 
df_fake['Label'] = 'Fake'
df_true['Label'] = 'Real'
df = pd.concat([df_fake, df_true], ignore_index=True)

df['title'], df['text'] = df['title'].fillna(''), df['text'].fillna('')
df['Full_Text'] = df['title'] + " " + df['text']
df['Final_Clean_Text'] = df['Full_Text'].apply(clean_news_text)

# Split and Vectorize 
X_train, X_test, y_train, y_test = train_test_split(
    df['Final_Clean_Text'], df['Label'], test_size=0.2, random_state=42, stratify=df['Label']
)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Train Passive Aggressive Classifier 
model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
model.fit(tfidf_train, y_train)

# Save models
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/fake_news_model.pkl')
joblib.dump(tfidf_vectorizer, '../models/tfidf_vectorizer.pkl')