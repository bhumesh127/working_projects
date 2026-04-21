import streamlit as st
import requests

st.title("RAG Chatbot")

user_input = st.text_input("Ask something:")

if st.button("Submit"):
    response = requests.post(
        "http://127.0.0.1:5000/query",
        json={"query": user_input}
    )

    st.write(response.json())