import re
from abc import abstractmethod
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text):
    """Cleans the input text by normalizing whitespace.

    It uses regular expressions to replace multiple whitespace characters with a single space and trims leading and trailing spaces.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing spaces
    text = text.strip()
    return text


class ChunkingStrategy:
    @abstractmethod
    def get_chunks(self, data):
        pass


class DefaultChunkingStrategy(ChunkingStrategy):
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size

    def get_chunks(self, data):
        chunks = []
        if isinstance(data, str):
            data = [data]
        for text in data:
            for i in range(0, len(text), self.chunk_size):
                max_size = min(self.chunk_size, len(text) - i)
                chunks.append(clean_text(text[i:i + max_size]))
        return chunks


class RecursiveSplitterChunkingStrategy(ChunkingStrategy):
    def __init__(self, chunk_size=256, overlap_size=30):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

    def get_chunks(self, data):
        chunks = []
        if isinstance(data, str):
            data = [data]
        texts = self.text_splitter.create_documents(data)
        for text in texts:
            chunks.append(clean_text(text.page_content))
        return chunks
