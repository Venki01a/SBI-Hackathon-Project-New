import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("hate_speech_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("🗣 Hate Speech Detection")

user_input = st.text_area("Enter text to check for hate speech:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
       
        import re, string, unicodedata
        def clean_text(text):
            text = str(text).lower()
            text = unicodedata.normalize("NFKD", text)
            text = re.sub(r"http\S+|www\S+", "", text)
            text = re.sub(r"@\w+|#", "", text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        label = "Hate Speech" if pred == 1 else "Non-Hate Speech"
        st.subheader(f"Prediction: {label}")