# Install required libraries (run once)
# pip install streamlit transformers torch

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set Streamlit page config
st.set_page_config(page_title="Text Generator with GPT-Neo 125M", layout="centered")

# Title of the app
st.title("📝 Text Generation with GPT-Neo 125M")
st.write("Enter a prompt, and the model will generate a coherent paragraph for you!")

# Load model and tokenizer (with caching for speed)
@st.cache_resource
def load_model():
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Text input from user
prompt = st.text_input("Enter your prompt here:")

# Additional settings
max_length = st.slider("Max Length", min_value=50, max_value=500, value=200, step=10)
temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
top_k = st.slider("Top-k Sampling", min_value=10, max_value=100, value=50, step=5)
top_p = st.slider("Top-p (nucleus sampling)", min_value=0.5, max_value=1.0, value=0.95, step=0.05)

# Generate button
if st.button("Generate Text"):
    if prompt:
        # Encode the input
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate outputs
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1
        )

        # Decode and show the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter a prompt to generate text!")



