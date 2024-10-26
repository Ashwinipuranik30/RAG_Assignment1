import streamlit as st
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wikipedia  # Use the correct Wikipedia library

# Load SentenceTransformer for embeddings
embedder = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # You can use 'distilgpt2' for a smaller model
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure pad_token is defined
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

# Function to search Wikipedia and retrieve relevant content
def get_relevant_wiki_content(query):
    try:
        # Search Wikipedia for the most relevant page
        search_results = wikipedia.search(query, results=1)

        if not search_results:
            return "No relevant Wikipedia content found."

        # Retrieve the content of the top search result
        page = wikipedia.page(search_results[0])

        # Split the page content into paragraphs
        paragraphs = page.content.split('\n\n')

        # Generate embeddings for the paragraphs and the query
        docs_embed = embedder.encode(paragraphs, normalize_embeddings=True)
        query_embed = embedder.encode(query, normalize_embeddings=True)

        # Compute similarities using dot product
        similarities = np.dot(docs_embed, query_embed.T)
        top_3_idx = np.argsort(similarities, axis=0)[-3:][::-1].tolist()

        # Collect the most relevant paragraphs
        most_relevant_docs = [paragraphs[idx] for idx in top_3_idx]
        return "\n\n".join(most_relevant_docs)

    except wikipedia.DisambiguationError as e:
        return f"Disambiguation error: {e.options}"
    except wikipedia.PageError:
        return "The relevant Wikipedia page does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

# Function to generate text using GPT-2
def call_gpt2(prompt, max_tokens=100):
    inputs = gpt2_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output = gpt2_model.generate(
        inputs['input_ids'],
        max_new_tokens=max_tokens,
        pad_token_id=gpt2_tokenizer.pad_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True).strip()

# Function to generate a response with RAG
def generate_response(query):
    # Retrieve relevant Wikipedia content based on the user query
    context = get_relevant_wiki_content(query)

    # Construct the prompt with the context and query
    prompt = f"""
    I have some information that might be relevant to answering a question.
    Here's the information:

    {context}

    Based on this information, please answer the following question:
    {query}

    If the information provided doesn't contain the answer, please say "I don't have enough information to answer this question."
    """
    return call_gpt2(prompt)

# Streamlit UI setup
st.title("Dynamic RAG Chatbot (GPT-2 + Wikipedia)")
st.write("Ask questions, and I'll find relevant information from Wikipedia to answer your query.")

# Conversation history storage
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# User input
user_query = st.text_input("Enter your question:")

# Generate a response when the user submits a query
if st.button("Send"):
    if user_query:
        response = generate_response(user_query)
        st.session_state.conversation_history.append(f"User: {user_query}")
        st.session_state.conversation_history.append(f"Bot: {response}")

# Display the conversation history
st.subheader("Conversation History")
for message in st.session_state.conversation_history:
    st.write(message)
