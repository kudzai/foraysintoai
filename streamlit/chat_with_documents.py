"""
This script utilizes Streamlit to create an interactive web interface for a semantic search-powered chatbot.
The chatbot can answer questions by searching through uploaded documents, providing answers based on the content
of those documents. It leverages the RunnerWithSemanticSearch class for processing user queries and generating
responses. The script supports uploading text or PDF documents, which are then indexed and searched by the chatbot
to find relevant information in response to user queries.

Key Features:
- Upload functionality for PDF and text files, allowing users to add content to the chatbot's knowledge base.
- A semantic search mechanism that queries the uploaded documents to find relevant answers.
- An interactive chat interface powered by Streamlit, enabling real-time conversation with the chatbot.
- Environment variables management with dotenv for loading configurations securely.

Usage:
To run the app, execute this script with Streamlit - 'streamlit run chat_with_documents.py' in your terminal or command prompt. Ensure that all required modules are installed and that
the .env file is configured with necessary environment variables. The app will start in a web browser, where
users can interact with the chatbot, upload documents, and query information based on the content of those documents.
"""
import streamlit as st
from dotenv import load_dotenv

from data_index import InMemoryDocumentIndex
from data_loader import DataLoader
from runner import RunnerWithSemanticSearch
from utils import show_chat_history

load_dotenv()


def main():
    runner = RunnerWithSemanticSearch(session_key='chat-with-with-documents')
    # Set up the Streamlit web interface
    st.title("Query an AI using semantic search")

    # Display all messages in the history
    show_chat_history(runner.get_history())
    document_index = initialize_document_index()

    # Handle input from the chat input box
    if question := st.chat_input():
        with st.expander(question, True):
            st.chat_message("human").write(question)
            with st.spinner("Thinking..."):
                response = runner.run(question, document_index)
                st.chat_message("ai").write(response.content)


def file_uploads(document_index):
    uploaded_files = st.file_uploader("Only pdf and txt formats are supported", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Processing {uploaded_file.name}...")
            document_index.add_to_index(uploaded_file)


def initialize_document_index():
    """Initialize or return the existing document index."""
    if 'document_index' not in st.session_state:
        data_loader = DataLoader()
        st.session_state.document_index = InMemoryDocumentIndex(data_loader)
    return st.session_state.document_index


if __name__ == "__main__":
    upload_files = st.checkbox(label="Check this if you want to upload files.", value=False)
    if upload_files:
        document_index = initialize_document_index()
        file_uploads(document_index)
    else:
        main()
