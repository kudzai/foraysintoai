import streamlit as st
from dotenv import load_dotenv  # For loading environment variables from a .env file

from runner import SimpleRunner
from utils import show_chat_history

# Load environment variables
load_dotenv()


def main():
    runner = SimpleRunner(session_key='chat-with-assistant-2')
    # Set up the Streamlit web interface
    st.title("Lets chat - Please ask me anything")

    # Display all messages in the history
    show_chat_history(runner.get_history())

    # Handle input from the chat input box
    if question := st.chat_input():
        with st.expander(question, True):
            st.chat_message("human").write(question)  # Display the human's question
            with st.spinner("Thinking..."):  # Show a loading spinner while processing
                response = runner.run(question)
                st.chat_message("ai").write(response.content)  # Display the AI's response


if __name__ == "__main__":
    main()
