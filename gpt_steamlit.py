import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # Options: 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure pad_token is defined (avoiding errors during text generation)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Streamlit UI setup
st.title("GPT-2 Text Generation with Streamlit")
st.write("Enter a prompt below, and GPT-2 will generate a continuation based on your input.")

# User input for text prompt
user_prompt = st.text_input("Enter your prompt:")

# Parameters for text generation (Optional: Adjustable sliders)
max_length = st.slider("Max Length", min_value=10, max_value=200, value=100)
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0)
top_k = st.slider("Top-k", min_value=1, max_value=100, value=50)
top_p = st.slider("Top-p (nucleus sampling)", min_value=0.1, max_value=1.0, value=0.95)

# Button to trigger text generation
if st.button("Generate Text"):
    if user_prompt:
        # Encode input prompt
        inputs = tokenizer.encode(user_prompt, return_tensors="pt")

        # Generate text using GPT-2
        output = model.generate(
            inputs, 
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode and display generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter a prompt to generate text.")

# Footer
st.markdown("Powered by **GPT-2** and **Streamlit**")
