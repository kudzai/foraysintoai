import os
from dotenv import load_dotenv
from autogen import ConversableAgent


# Assumes there is a .env file with the api key
load_dotenv()

llm_config = {"model": "gpt-3.5-turbo"}


teacher = ConversableAgent(
    name="maestro",
    system_message= 
    "Your name is Isaac and you are a physics teacher."
    "The student will ask questions asking you to explain. You only answer physics questions.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

student = ConversableAgent(
    name="student",
    system_message=
    "Your name is Joe and you are a student. "
    "You don't know Newton's laws of motion, and you want the teacher to help you with that."
    "You will the teacher questions so he can explain."
    "When you finally understand and want to end the conversation, say 'I understand. Thank you.'",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "I understand. Thank you" in msg["content"] or "Goodbye" in msg["content"],
)

if __name__ == "__main__":
    chat_result = student.initiate_chat(
        recipient=teacher,
        message="Good morning, can I please have some help with Newton's laws of motion?"
    )