import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore") 
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("FATAL ERROR: data.csv not found. Please ensure your dataset is named 'data.csv' and is in the same folder.")
    exit()

df['Body'] = df['Body'].fillna('')
df.columns = ['URLs', 'Headline', 'Body', 'Label'] 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    return text.strip()

df['clean_text'] = df['Body'].apply(clean_text)


X = df['clean_text']
y = df['Label']


vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    stop_words='english'
)

X_features = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)


model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, 'fake_news_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("\n-------------------------------------------")
print("✅ Mission 1 Complete: Model Files SAVED!")
print("   (fake_news_model.joblib and tfidf_vectorizer.joblib)")
print("-------------------------------------------")
