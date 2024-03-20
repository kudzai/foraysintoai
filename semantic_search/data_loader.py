import os

from chunking import DefaultChunkingStrategy
from utils import read_text_file, parse_pdf

class DataLoader:
    """
    DataLoader is a class for loading and batching data from text and PDF files within a specified directory.

    Attributes:
        directory (str): The directory path from which to load files.
        chunker (DefaultChunkingStrategy): The strategy for chunking loaded data.
            Defaults to an instance of DefaultChunkingStrategy.
        batch_size (int): The number of chunks to include in each batch. Defaults to 512.

    Methods:
        load(): Generator method that yields content from text and PDF files in the directory.
        get_chunks(): Retrieves chunks from the loaded files using the chunker strategy.
        get_chunk_batches(): Yields batches of chunks according to the specified batch size.
        __iter__(): Returns an iterator for chunk batches, allowing the DataLoader to be used in a for-loop.
    """

    def __init__(self, directory, batch_size=512, chunker=DefaultChunkingStrategy()):
        """
        Initializes the DataLoader with a directory, optional batch size, and chunking strategy.

        Parameters:
            directory (str): The directory path from which to load files.
            batch_size (int): The number of chunks to include in each batch. Defaults to 512.
            chunker (DefaultChunkingStrategy): The strategy for chunking loaded data.
                Defaults to an instance of DefaultChunkingStrategy.
        """
        self.directory = directory
        self.chunker = chunker
        self.batch_size = batch_size

    def load(self):
        """
        Generator that walks through the directory, loading content from each text or PDF file found.

        Yields:
            The content of each file, processed based on the file extension (.txt or .pdf).
        """
        for root, _, files in os.walk(self.directory):
            for file in files:
                print(f"Loading file: {os.path.join(root, file)}")
                file_path = os.path.join(root, file)
                if file_path.endswith('.txt'):
                    yield read_text_file(file_path)
                elif file_path.endswith('.pdf'):
                    yield parse_pdf(file_path)

    def get_chunks(self):
        """
        Uses the chunker strategy to get chunks from the loaded files.

        Returns:
            A generator yielding chunks of data as defined by the chunker strategy.
        """
        return self.chunker.get_chunks(self.load())

    def get_chunk_batches(self):
        """
        Yields batches of data chunks, with each batch containing up to the specified batch size.

        If the final batch has fewer chunks than the batch size, it will still be yielded.

        Yields:
            A list of chunks, where each list is a batch of data.
        """
        chunks = []
        for chunk in self.get_chunks():
            chunks.append(chunk)
            if len(chunks) == self.batch_size:
                yield chunks
                chunks = []

        # Yield any remaining chunks as the final batch
        if len(chunks) > 0:
            yield chunks

    def __iter__(self):
        """
        Allows the DataLoader to be used as an iterator, yielding chunk batches.

        Returns:
            An iterator for the get_chunk_batches method.
        """
        return self.get_chunk_batches()
