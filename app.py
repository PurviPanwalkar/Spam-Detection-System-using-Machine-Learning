import streamlit as st
import pickle
import re

model = pickle.load(open('spam_model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    return text

st.title("Spam Detection")

msg = st.text_area("Enter Message")

if st.button("Check"):
    msg = clean_text(msg)
    vec = vectorizer.transform([msg])
    result = model.predict(vec)

    if result[0] == 1:
        st.error("Spam 🚫")
    else:
        st.success("Ham ✅")
