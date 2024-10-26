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
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Ensure pad_token is defined
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

# Initialize LLaMA3 API endpoint
LLAMA_API_URL = "http://localhost:11434/api/chat"

### --- Helper Functions --- ###

# Search Wikipedia and retrieve relevant content
def get_relevant_wiki_content(query):
    try:
        search_results = wikipedia.search(query, results=1)

        if not search_results:
            return "No relevant Wikipedia content found."

        page = wikipedia.page(search_results[0])
        paragraphs = page.content.split('\n\n')

        # Generate embeddings for paragraphs and the query
        docs_embed = embedder.encode(paragraphs, normalize_embeddings=True)
        query_embed = embedder.encode(query, normalize_embeddings=True)

        # Compute similarities and find top 3 relevant paragraphs
        similarities = np.dot(docs_embed, query_embed.T)
        top_3_idx = np.argsort(similarities, axis=0)[-3:][::-1].tolist()

        return "\n\n".join([paragraphs[idx] for idx in top_3_idx])
    except wikipedia.DisambiguationError as e:
        return f"Disambiguation error: {e.options}"
    except wikipedia.PageError:
        return "The relevant Wikipedia page does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

# Generate text using GPT-2
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

# Call the LLaMA API
def call_llama(prompt, temperature=0.7, max_tokens=300):
    payload = {
        "model": "llama3.2",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(LLAMA_API_URL, json=payload, stream=True, timeout=30)
        response.raise_for_status()

        chunks = []
        for chunk in response.iter_lines():
            if chunk:
                try:
                    chunk_data = json.loads(chunk.decode("utf-8").strip())
                    chunks.append(chunk_data)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed chunk: {chunk}, Error: {e}")

        return "".join(chunk.get("message", {}).get("content", "") for chunk in chunks).strip()
    except requests.exceptions.Timeout:
        return "Error: LLaMA API timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# RAG response generation (Wikipedia + LLM)
def generate_rag_response(query, use_gpt2=True):
    context = get_relevant_wiki_content(query)
    prompt = f"""
    I have some information that might be relevant to answering a question.
    Here's the information:

    {context}

    Based on this information, please answer the following question:
    {query}

    If the information provided doesn't contain the answer, please say "I don't have enough information to answer this question."
    """
    if use_gpt2:
        return call_gpt2(prompt)
    else:
        return call_llama(prompt)

# Non-RAG response generation (LLM only)
def generate_non_rag_response(query, use_gpt2=True):
    if use_gpt2:
        return call_gpt2(query)
    else:
        return call_llama(query)

### --- Streamlit UI Setup --- ###

st.title("Dynamic Chatbot (RAG + LLM Modes)")
st.write("Ask questions and choose between using RAG mode or a standalone LLM.")

# Dropdown to select mode (RAG or Non-RAG) and LLM (GPT-2 or LLaMA)
mode = st.radio("Choose mode:", ["RAG (Wikipedia + LLM)", "LLM Only"])
llm_model = st.selectbox("Choose LLM model:", ["GPT-2", "LLaMA"])

# Conversation history storage
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# User input
user_query = st.text_input("Enter your question:")

# Generate a response when the user submits a query
if st.button("Send"):
    if user_query:
        use_gpt2 = llm_model == "GPT-2"

        # Route the query based on the selected mode
        if mode == "RAG (Wikipedia + LLM)":
            response = generate_rag_response(user_query, use_gpt2)
        else:
            response = generate_non_rag_response(user_query, use_gpt2)

        # Store the conversation history
        st.session_state.conversation_history.append(f"User: {user_query}")
        st.session_state.conversation_history.append(f"Bot: {response}")

# Display the conversation history
st.subheader("Conversation History")
for message in st.session_state.conversation_history:
    st.write(message)
