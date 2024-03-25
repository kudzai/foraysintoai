from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

SIMPLE_SYSTEM_PROMPT = "You are a chatbot having a conversation with a human."

SYSTEM_PROMPT_WITH_CONTEXT = '''
You are a helpful assistant that can only reference material from a knowledge base.
You do not like using any of your general knowledge.
You may only use information prefixed by in the context.
If the question cannot be answered only using information from context, say instead "I'm sorry I cannot answer that."
Context: {context}
'''.strip()


class BaseRunner:
    def run(self, question):
        pass

    def get_history(self):
        pass


class SimpleRunner(BaseRunner):
    """
    A chatbot runner implementation that handles basic chat interactions without
    the need for context or external data sources. It uses a predefined model and
    a simple system prompt to engage in conversation with a user.

    Initialization Parameters:
        session_key (str): A unique key for the chat session to manage history.
        model (str): The model identifier to use for generating chat responses.
        temperature (float): The creativity parameter for the model's responses.

    Methods:
        get_history(): Retrieves the chat message history for the current session.
        run(question): Executes the chatbot logic for a given question, integrating
                       any existing chat history into the conversation.
    """
    def __init__(self, session_key, model='gpt-4-0125-preview', temperature=0.7):
        self.model = model
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SIMPLE_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name='history'),
                ("human", "{question}")
            ]
        )
        chain = prompt | ChatOpenAI(model=model, temperature=temperature)

        self.history = StreamlitChatMessageHistory(key=session_key)
        if len(self.history.messages) == 0:
            self.history.add_ai_message('How can I help you today?')

        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.history,
            input_messages_key="question",
            history_messages_key="history"
        )

    def get_history(self):
        return self.history.messages

    def run(self, question):
        config = {"configurable": {"session_id": "any"}}
        return self.chain_with_history.invoke({"question": question, "context": ""}, config)


class RunnerWithSemanticSearch(BaseRunner):
    """
    Extends SimpleRunner by incorporating semantic search capabilities. This runner
    uses document search to find relevant information to a user's question and
    integrates this information into the chat context for more informed responses.

    Initialization Parameters:
        session_key (str): A unique key for the chat session to manage history.
        model (str): The model identifier for generating chat responses.
        temperature (float): The creativity parameter for the model's responses.
        max_documents (int): The maximum number of documents to retrieve for context.

    Methods:
        get_history(): Retrieves the chat message history for the current session.
        run(question, document_index): Extends the run method to include document
                                       search before executing the chatbot logic.
    """
    def __init__(self, session_key, model='gpt-4-0125-preview', temperature=0.7, max_documents=10):
        self.model = model
        self.max_documents = max_documents
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_WITH_CONTEXT),
                MessagesPlaceholder(variable_name='history'),
                ("human", "{question}")
            ]
        )

        chain = prompt | ChatOpenAI(model=model, temperature=temperature)

        self.history = StreamlitChatMessageHistory(key=session_key)
        if len(self.history.messages) == 0:
            self.history.add_ai_message('How can I help you today?')

        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.history,
            input_messages_key="question",
            history_messages_key="history"
        )

    def get_history(self):
        return self.history.messages

    def run(self, question, document_index):
        config = {"configurable": {"session_id": "any"}}
        docs = document_index.query(question, k=self.max_documents)
        return self.chain_with_history.invoke({"question": question, "context": docs}, config)
