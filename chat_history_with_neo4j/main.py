# Import required modules and components from various packages
import streamlit as st
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.chat_history import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from dotenv import load_dotenv
import os

# Load environment variables from a .env file into the system environment
load_dotenv()

# Retrieve Neo4j database connection details from environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

# Define the local large language model to be used
local_llm = "mistral:7b-instruct"

# Set the programming language for the chat session
programming_language = "java"
chat_history_session_id = f"session_{programming_language}"

# Define a system prompt that explains the AI's role
SYSTEM_PROMPT = """You are an expert programmer in the programming language specified by the user. Your task is to 
carefully read the user's question, and provide a clear answer with step-by-step explanation. Try to give alternative 
implementations if possible."""

# Setup the chat prompt template with system prompt and placeholders for chat history
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Programming Language: {language}\nQuestion: {input}")
])

# Initialize the Neo4j chat message history handler
messages_history = Neo4jChatMessageHistory(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    session_id=chat_history_session_id
)


# Function to generate a response using the chain of chat operations
def generate_response(chain, input_text, chosen_language=programming_language):
    response = chain.invoke(
        {"input": input_text, "language": chosen_language, "chat_history": messages_history.messages.reverse() or []})
    messages_history.add_messages([HumanMessage(input_text), AIMessage(response)])
    show_message(response, "assistant")


# Helper function to display messages in the Streamlit interface
def show_message(message, role):
    with container.chat_message(role):
        container.markdown(message)


# Function to display all messages from the chat history
def show_messages():
    for message in messages_history.messages:
        if isinstance(message, HumanMessage):
            show_message(message.content, "user")
        else:
            show_message(message.content, "assistant")


# Initialize the language model and chain of operations for parsing and handling responses
llm = ChatOllama(temperature=0, max_tokens=1024, model=local_llm)
chain = chat_prompt | llm | StrOutputParser()

# Set Streamlit page configuration
st.set_page_config(page_title='My Language Tutor')
intro = f"""
Hi, I am your programming assistant. 
I can help you with any {programming_language} programming questions you have.
"""
st.info(intro)

# Create a container in Streamlit for managing chat interactions
container = st.container(border=True)

# Display all previous messages
show_messages()

# Input loop for new chat messages
if prompt := st.chat_input("How can I help?:"):
    show_message(prompt, "user")
    with st.spinner("Generating response ... please wait"):
        generate_response(chain, prompt)
