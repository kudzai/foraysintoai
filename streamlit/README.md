## Streamlit demo apps
This contains 2 demo apps using streamlit UI - both will open a browser window for the interaction.
1. Simple conversation with an AI assistant - ask anything.
Run from the command line using:
```bash 
streamlit run multi_turn_chat.py
```

2. Querying documents using FAISS vector store
Run using:
```bash 
streamlit run chat_with_documents.py
```
This allows to upload documents in pdf or txt format, and then run queries against them. For the example file in data folder, you could try the following queries:
- "What is the state of investment in the renewable energy sector?"
- "Comment on the impact on FDI of national security regulations"

## Installation
Before running make sure you have the following packages installed:
```bash
pip install langchain langchain-openai pymupdf streamlit langchain-community python-dotenv faiss-cpu sentence-transformers langchain-text-splitters
```
or just install from the requirements.txt file.
```bash
pip install -r requirements.txt
```
## Environment variables
Rename .env.sample to .env, and add your OpenAI API key to the .env file.

## Docs
For further exploration, please refer to the following documentation:
- https://python.langchain.com/docs/integrations/memory/streamlit_chat_message_history
- https://streamlit.io/generative-ai
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2