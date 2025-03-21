
import streamlit as st
from app import extractive_summarization  # Import your summarization function


st.title("Extractive Summarization Tool")


input_text = st.text_area("Enter the text to summarize:", height=200)

num_sentences = st.number_input("Number of sentences in the summary:", min_value=1, value=3)


if st.button("Summarize"):
    if input_text.strip():
        summary = extractive_summarization(input_text, num_sentences)
        st.subheader("Extractive Summary:")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")