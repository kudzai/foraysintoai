from openai import OpenAI
from sentence_transformers import SentenceTransformer

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
