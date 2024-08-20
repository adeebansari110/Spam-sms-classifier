import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # clonning
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title("Email/Sms Spam Classifier")

input_sms=st.text_input("Enter the message")
if st.button('🔍 Predict'):
    # Display styled text when the button is clicked
   # st.markdown("<h2 style='color: blue;'>Prediction in progress...</h2>", unsafe_allow_html=True)
    #1 Preprocess
    transformed_sms=transform_text(input_sms)
    #2 vectorize
    vector_input=tfidf.transform([transformed_sms])
    #predict
    result=model.predict(vector_input)[0]
    #4. display
    if result == 1:
        st.markdown("<h2 style='color: red;'>Spam</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green;'>Not Spam</h2>", unsafe_allow_html=True)
