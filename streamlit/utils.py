import streamlit as st


# Show chat history in a more readable format
def show_chat_history(history):
    for i in range(1, len(history), 2):
        question = history[i] if i < len(history) else None
        answer = history[i + 1] if i + 1 < len(history) else None

        if question and answer:
            with st.expander(question.content, expanded=False):
                st.chat_message(question.type).write(question.content)
                st.chat_message(answer.type).write(answer.content)


def show_chat_history_simple(history):
    for msg in history:
        st.chat_message(msg.type).write(msg.content)
