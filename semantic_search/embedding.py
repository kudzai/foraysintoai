from openai import OpenAI
from sentence_transformers import SentenceTransformer

from utils import OPENAI_API_KEY

class OpenaiEmbedding:
    """
    A class for generating embeddings using OpenAI's API.
    """
    def __init__(self, client=OpenAI(api_key=OPENAI_API_KEY)):
        """
        Initializes the OpenaiEmbedding object with the OpenAI client,
        model name, embedding dimensions, and a custom name for the model.

        Parameters:
        - client (OpenAI): The OpenAI client instance configured with an API key.
        """
        self.model_name = "text-embedding-3-large"  # The name of the OpenAI embedding model to use
        self.client = client
        self.dimensions = 1024  # The dimensionality of the generated embeddings
        self.name = "OpenAI"  # A custom name for the embedding model

    def get_embedding(self, text):
        """
        Generates an embedding for the given text using OpenAI's API.

        Parameters:
        - text (str): The input text to generate an embedding for.

        Returns:
        - The embedding vector as a list of floats.
        """
        text = text.replace('\n', ' ')
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
            dimensions=self.dimensions
        )
        return response.data[0].embedding


class OpenSourceEmbedding:
    """
    A class for generating embeddings using the Sentence Transformers library.
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initializes the OpenSourceEmbedding object with the specified model.

        Parameters:
        - model_name (str): The name of the Sentence Transformer model to use.
        """
        self.model = SentenceTransformer(model_name)  # Load the Sentence Transformer model
        self.dimensions = 384  # The dimensionality of the generated embeddings
        self.name = model_name.split('/')[1]  # Extract the model name for identification

    def get_embedding(self, text):
        """
        Generates an embedding for the given text using the loaded Sentence Transformer model.

        Parameters:
        - text (str): The input text to generate an embedding for.

        Returns:
        - The embedding vector as a NumPy array.
        """
        text = text.replace('\n', ' ')
        # Encode the text into an embedding and convert it to a NumPy array
        return self.model.encode(text, convert_to_numpy=True)
