import datetime
import hashlib
from abc import abstractmethod
from pinecone import Pinecone, ServerlessSpec, PodSpec
from tqdm import tqdm
import faiss
import numpy as np
from embedding import OpenSourceEmbedding
from utils import PINECONE_API_KEY

class DataIndex:
    """
    Abstract base class for data indexing and querying.
    """

    @abstractmethod
    def query(self, query, k=5):
        """
        Query the index for the top k most similar items to the given query.

        Args:
            query (str): The query text.
            k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            list: A list of the top k most similar items to the query.
        """
        pass

    @abstractmethod
    def get_embeddings(self, data):
        """
        Get the embeddings for the given data.

        Args:
            data (list): A list of text data.

        Returns:
            numpy.ndarray: A numpy array of embeddings for the given data.
        """
        pass

class InMemoryIndex(DataIndex):
    """
    An in-memory index for data indexing and querying.
    """

    def __init__(self, loader, embed=OpenSourceEmbedding()):
        """
        Initialize the InMemoryIndex.

        Args:
            loader (generator): A generator that yields batches of text data.
            embed (OpenSourceEmbedding, optional): The embedding model to use. Defaults to OpenSourceEmbedding().
        """
        self.index = None
        self.loader = loader
        self.embed = embed
        self.build_index()

    def build_index(self):
        """
        Build the in-memory index from the data loader.
        """
        self.content_chunks = []
        self.index = None
        for chunk_batch in tqdm(self.loader):
            embeddings = self.get_embeddings(chunk_batch)
            if self.index is None:
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(embeddings)
            self.content_chunks.extend(chunk_batch)

    def get_embeddings(self, data):
        """
        Get the embeddings for the given data.

        Args:
            data (list): A list of text data.

        Returns:
            numpy.ndarray: A numpy array of embeddings for the given data.
        """
        embedding_list = [self.embed.get_embedding(text) for text in data]
        return np.array(embedding_list)

    def query(self, query, k=5):
        """
        Query the index for the top k most similar items to the given query.

        Args:
            query (str): The query text.
            k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            list: A list of the top k most similar items to the query.
        """
        embedding = self.get_embeddings([query])[0]
        embedding_array = np.array([embedding])
        _, indices = self.index.search(embedding_array, k)
        return [self.content_chunks[i] for i in indices[0]]

class PineconeIndex(DataIndex):
    """
    A Pinecone index for data indexing and querying.
    """

    def __init__(self, loader, embed=OpenSourceEmbedding(), index_name="semantic-search"):
        """
        Initialize the PineconeIndex.

        Args:
            loader (generator): A generator that yields batches of text data.
            embed (OpenSourceEmbedding, optional): The embedding model to use. Defaults to OpenSourceEmbedding().
            index_name (str, optional): The name of the Pinecone index. Defaults to "semantic-search".
        """
        self.loader = loader
        self.embed = embed
        self.index_name = f"{embed.name}-{index_name}".lower()
        self.content_chunks = []
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.initialize_pinecone()
        self.build_index()

    def initialize_pinecone(self):
        """
        Initialize the Pinecone index.
        """
        # Check if the index exists
        if self.index_name not in self.pc.list_indexes().names():
            # Create a new index if it does not exist
            print(f"Creating index {self.index_name}...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embed.dimensions,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
        else:
            print(f"Index {self.index_name} already exists.")
        self.index = self.pc.Index(name=self.index_name)

    def build_index(self):
        """
        Build the Pinecone index from the data loader.
        """
        for chunk_batch in tqdm(self.loader):
            embeddings = self.get_embeddings(chunk_batch)
            vectors = [{
                "id": id_hash(text),
                "values": embedding,
                "metadata": {"text": text, "date_upserted": datetime.datetime.now()}
            } for text, embedding in zip(chunk_batch, embeddings)]
            self.index.upsert(vectors=vectors)
            self.content_chunks.extend(chunk_batch)

    def get_embeddings(self, data):
        """
        Get the embeddings for the given data.

        Args:
            data (list): A list of text data.

        Returns:
            list: A list of embeddings for the given data.
        """
        embedding_list = [self.embed.get_embedding(text) for text in data]
        return embedding_list

    def query(self, query_text, top_k=5):
        """
        Query the index for the top k most similar items to the given query.

        Args:
            query_text (str): The query text.
            top_k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            list: A list of the top k most similar items to the query.
        """
        query_embedding = self.get_embeddings([query_text])[0]
        query_results = self.index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
        results = [match.metadata['text'] for match in query_results["matches"]]
        return results

def id_hash(s):
    """
    Return the MD5 hash of the input string as a hexadecimal string.

    Args:
        s (str): The input string.

    Returns:
        str: The MD5 hash of the input string as a hexadecimal string.
    """
    return hashlib.sha1(s.encode()).hexdigest()