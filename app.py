import streamlit as st
import pickle
import string
import nltk

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
stopwords.words('english')
ps=PorterStemmer()
stop_words=set(stopwords.words('english'))
punctuation_set=set(string.punctuation)

def transform_text(text):
    text = text.lower()
    text=nltk.word_tokenize(text)

    filtered_text=[
        ps.stem(i) for i in text 
        if i.isalnum() and i not in stop_words and i not in punctuation_set
    ]
    return " ".join(filtered_text)


st.title("Email/SMS Classifier")


input_sms=st.text_input("Paste your SMS/Email here")

#1 PREPROCESS

transformed_text=transform_text(input_sms)
#2 VECTORIZE
vector_input=tfidf.transform([transformed_text])
#3 PREDICT
result=model.predict(vector_input)[0]
#4 DISPLAY

# if st.text_input is None:
#     st.header("Please Enter or Paste any message")

if result==1:
#     # st.header("Spam")
    st.markdown("""
    <h1 style='color: red;'>Spam</h1>
    """, unsafe_allow_html=True)

else:
    # st.header("Not Spam")
    st.markdown("""<h1 style='color: green;'>Not Spam</h1> """, unsafe_allow_html=True)
