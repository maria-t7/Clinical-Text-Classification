import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download required nltk data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = MEDICAL_HEADERS.sub(' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
        if word not in stop_words and (len(word) > 2 or (word.isdigit() and len(word) >= 2))
    ]
    return ' '.join(tokens)

st.title("Clinical Text Classifier")
st.write("Enter a medical transcription to predict the medical specialty.")

text_input = st.text_area("Medical Transcription", height=200)

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"Predicted Specialty: **{prediction}**")

