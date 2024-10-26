import streamlit as st
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import wikipedia  # Use the correct Wikipedia library

# Initialize LLaMA3 API endpoint
LLAMA_API_URL = "http://localhost:11434/api/chat"

# Load SentenceTransformer for embeddings
embedder = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

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

# Function to call the LLaMA API
def call_llama(model_name, prompt, temperature=0.7, max_tokens=300):
    url = LLAMA_API_URL
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)
        response.raise_for_status()

        # Accumulate all response chunks
        chunks = []
        for chunk in response.iter_lines():
            if chunk:
                decoded_chunk = chunk.decode("utf-8").strip()
                try:
                    chunk_data = json.loads(decoded_chunk)
                    chunks.append(chunk_data)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed chunk: {decoded_chunk}, Error: {e}")

        # Combine the content from valid chunks
        full_content = "".join(
            chunk.get("message", {}).get("content", "") for chunk in chunks
        )
        return full_content.strip()

    except requests.exceptions.Timeout:
        return "Error: LLaMA API timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Function to generate a response with RAG
def generate_response(query, model_name):
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
    return call_llama(model_name, prompt)

# Streamlit UI setup
st.title("Dynamic RAG Chatbot (LLaMA + Wikipedia)")
st.write("Ask questions, and I'll find relevant information from Wikipedia to answer your query.")

# Conversation history storage
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# User input
user_query = st.text_input("Enter your question:")

# Generate a response when the user submits a query
if st.button("Send"):
    if user_query:
        model_name = "llama3.2"
        response = generate_response(user_query, model_name)
        st.session_state.conversation_history.append(f"User: {user_query}")
        st.session_state.conversation_history.append(f"Bot: {response}")

# Display the conversation history
st.subheader("Conversation History")
for message in st.session_state.conversation_history:
    st.write(message)
