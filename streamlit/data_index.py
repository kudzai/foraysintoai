import faiss
import numpy as np
from tqdm import tqdm

from embedding import OpenSourceEmbedding


class InMemoryDocumentIndex:
    def __init__(self, data_loader, embedding_model=OpenSourceEmbedding()):
        self.index = None
        self.embedding_model = embedding_model
        self.content_chunks = []
        self.data_loader = data_loader
        self.metadata =[]

    def add_to_index(self, file, metadata=None):
        if metadata is None:
            metadata = {"source": file.name}
            source = file.name
        else:
            source = metadata["source"]

        print(f"metadata: {self.metadata}")
        for data in self.metadata:
            if data["source"] == source:
                print(f"File {file.name} already in index.")
                return

        print(f"Indexing file {file.name}...")
        self.metadata.append(metadata)
        print(f"after metadata: {self.metadata}")

        for chunk_batch in tqdm(self.data_loader.get_chunk_batches(file)):
            embeddings = self.get_embeddings(chunk_batch)
            if self.index is None:
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(embeddings)
            self.content_chunks.extend(chunk_batch)
        print(f"Done indexing file {file.name}.")

    def get_embeddings(self, data):
        """
        Get the embeddings for the given data.

        Args:
            data (list): A list of text data.

        Returns:
            numpy.ndarray: A numpy array of embeddings for the given data.
        """
        embedding_list = [self.embedding_model.get_embedding(text) for text in data]
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
