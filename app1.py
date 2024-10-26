import streamlit as st
import subprocess
import requests
import json

# Function to interact with the LLM API
def call_llm(model_name, user_input, temperature=0.7, max_tokens=20):
    url = "http://localhost:11434/api/chat"  # Ollama API URL
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": user_input.strip()}
        ]
    }

    # Send the request and get the response
    response = requests.post(url, json=payload, stream=True)
    full_response = ""
    # Handle streamed response from the model
    for chunk in response.iter_lines():
        if chunk:
            chunk_data = json.loads(chunk.decode("utf-8"))
            full_response += chunk_data.get("message", {}).get("content", "")
            if chunk_data.get("done", False):
                break
    return full_response

# Function to preprocess user input
def preprocess_query(query):
    return query.lower().strip()

# Function to post-process LLM response
def postprocess_response(response):
    return response.strip()

# Function to get a response from the LLM
def get_llm_response(model_name, user_query):
    preprocessed_query = preprocess_query(user_query)
    raw_response = call_llm(model_name, preprocessed_query)
    return postprocess_response(raw_response)

# Streamlit UI Configuration for the Domain-Specific Chatbot
DOMAIN = "historical facts"  # This can be changed to any domain like medical advice, fitness, etc.

st.title(f"{DOMAIN.capitalize()} Chatbot")
st.write(f"Ask questions about {DOMAIN}!")

# Conversation history storage
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# User input
user_input = st.text_input(f"Enter your question related to {DOMAIN}:")

# If the user submits a query, call the LLM
if st.button("Send"):
    if user_input:
        # Prepend domain-specific context to the query
        domain_prompt = f"You are a chatbot specialized in {DOMAIN}. Please provide a helpful answer. Question: {user_input}"

        # Call the backend to get the response from the LLM
        model_name = "llama3.2"  # or another model if needed
        response = get_llm_response(model_name, domain_prompt)

        # Store the conversation history
        st.session_state.conversation_history.append(f"User: {user_input}")
        st.session_state.conversation_history.append(f"Bot: {response}")

# Display conversation history
st.subheader("Conversation History")
for message in st.session_state.conversation_history:
    st.write(message)
