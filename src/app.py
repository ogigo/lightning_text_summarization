import inference
import lightning as pl
import torch
import transformers
from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer
from model import SummaryModel
import streamlit as st

# Streamlit app title and description
st.title("Text Summarization App")
st.write("Enter your text below, and we'll generate a summary for you!")

# User input for text
user_input = st.text_area("Enter your text here:")

# Summarization button
if st.button("Summarize"):
    if not user_input:
        st.warning("Please enter some text to summarize.")
    else:
        summary=inference.generate_summary(user_input)

        st.subheader("Summary:")
        st.write(summary)