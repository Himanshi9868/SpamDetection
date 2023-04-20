import streamlit as st

import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def transform_text(text):
    #lower case
    text = text.lower()
    #tokenize
    text = nltk.word_tokenize(text)
    #remove special characters
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    #remove stopwords and punctuations
    text = y[:]
    y.clear()
    for i in text:
        import string
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    #stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfid = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    #1.Preprocessing
    transformed_sms = transform_text(input_sms)

    #2.Vectorization
    vector_sms = tfid.transform([transformed_sms])

    #3.Prediction
    result = model.predict(vector_sms)[0]

    #4.Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")



