# RAG_Assignment1



# **RAG-Enhanced Chatbot with LLaMA3 and GPT-2**

## **Overview**
This project demonstrates a chatbot using the **Retrieval-Augmented Generation (RAG)** technique with **LLaMA3**. It allows users to ask questions, retrieve relevant information from **Wikipedia**, and generate context-aware responses. The chatbot runs entirely on a **local machine**, leveraging **LLaMA3 via Ollama** and **GPT-2 model**, with an easy-to-use **Streamlit UI**.

---

## **Features**
- **RAG System**: Retrieves relevant content from Wikipedia based on user queries.
- **Multiple LLMs**: Switch between **LLaMA3** and **GPT-2** models for generating responses.
- **Interactive User Interface**: Built with **Streamlit** for real-time user interaction.
- **Local Execution**: All models run locally using **Ollama**.
- **Parameter Customization**: Users can adjust temperature, max tokens, top-k, and top-p sampling parameters.
- **Conversation History**: Keeps track of user-bot interactions during each session.

---

## **Prerequisites**
- **Python** 3.8+
- **Ollama Installed**: To run LLaMA3 models locally.

**Install Streamlit and Required Packages:**
```bash
pip install streamlit transformers sentence-transformers wikipedia requests
```



---

## **Set Up Instructions**

- **Clone the Repository
- **git clone https://github.com/Ashwinipuranik30/RAG_Assignment1.git
- ** Set Up a Virtual Environment**
- **Install Dependencies**
- **pip install -r requirements.txt**
- **Download the LLaMA3 Model**
```bash
ollama pull llama3
```
- **Start the Ollama Service**
- **Run the Streamlit App**
**streamlit run rag_llama.py (RAG mode using llama3 LLM)**
**streamlit run rag_gpt2.py (RAG mode using GPT2 LLM)**
**streamlit run rag_gpt_llama.py (RAG mode using llama3 & GPT2 LLMs)**
**streamlit run gpt_streamlit.py (Standalone Gpt2 model)**
**streamlit run app1.py (Standalone llama3 model)**

-**Open the Streamlit App**
**Visit http://localhost:8501 in your browser.**
**Enter a Query:**
**Type your question in the text input box and click Send.**
**Adjust Parameters:**


##**Technologies Used**
- Python 3.8+
- Streamlit: Interactive UI for web-based interfaces.
- Ollama: To run LLaMA3 models locally.
- Transformers Library: For GPT-2 text generation.
- Sentence Transformers: For embedding Wikipedia content.
- Wikipedia Library: To fetch relevant content for queries.


##**Challenges Faced**
-**Handling Multi-Chunk JSON Responses**:
Implemented logic to accumulate and combine partial JSON responses from LLaMA3.
-**Connection Errors with Streamlit**:
Ensured Ollama service was properly running to avoid connection issues.
-**Disambiguation Handling with Wikipedia**:
Added error handling for disambiguation and page-not-found scenarios.




