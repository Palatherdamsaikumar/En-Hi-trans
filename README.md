# 🌍 AI Language Translator (English → French)

This project is a mini language translation tool that converts English text into French using a transformer-based model from Hugging Face. It demonstrates the power of pretrained models for sequence-to-sequence tasks like translation.

---

 Project Highlights

**Input**: English text
**Model**: Helsinki-NLP/opus-mt-en-fr
**Output**: Translated French text
Built and tested on **Google Colab**
Optionally deployable using **Streamlit** as a web app


Technologies Used

- Python
- Hugging Face Transformers
- MarianMT Model (Opus-MT)
- PyTorch
- SentencePiece
- Google Colab
  Streamlit for deployment

How to Run
Install dependencies:
bash
pip install transformers torch sentencepiece streamlit



code :
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

st.title("Language Translator - English to French")

model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = st.text_input("Enter English text:")
if text:
    tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
    translation = model.generate(**tokens)
    output = tokenizer.batch_decode(translation, skip_special_tokens=True)
    st.write("Translated:", output[0])

