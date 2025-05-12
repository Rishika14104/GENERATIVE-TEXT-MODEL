import streamlit as st
from transformers import GPTNeoForCausalLM, AutoTokenizer
import torch

# Load pre-trained GPT-Neo 1.3B model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Streamlit app layout
st.title("GPT-Neo 1.3B Text Generator")
st.subheader("Generate coherent text with GPT-Neo 1.3B model")

# User input field for the prompt
prompt = st.text_area("Enter your topic or prompt:", "The impact of climate change on agriculture")

# Temperature and max length sliders
temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7)
max_length = st.slider("Max length of generated text", 50, 500, 150)

# Button to generate text
if st.button("Generate Text"):
    with st.spinner("Generating..."):
        try:
            # Encode the input prompt and generate text
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(
                **inputs, 
                max_length=max_length, 
                temperature=temperature, 
                num_return_sequences=1, 
                no_repeat_ngram_size=2
            )

            # Decode the generated text and display it
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            st.markdown("### Generated Text:")
            st.write(generated_text)
        except Exception as e:
            st.error(f"Error: {e}")




