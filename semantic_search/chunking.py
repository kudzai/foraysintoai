import re
import spacy


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


class DefaultChunkingStrategy:
    """Splits text into chunks of a specified size without considering the textual content.

    Attributes:
        chunk_size (int): The maximum size of each text chunk.
    """

    def __init__(self, chunk_size=512):
        """Initializes the DefaultChunkingStrategy with a specified chunk size.

        Args:
            chunk_size (int): The maximum size of each text chunk. Defaults to 512.
        """
        self.chunk_size = chunk_size

    def get_chunks(self, data):
        """Splits the input text into chunks of up to chunk_size characters.

        Args:
            data (str or list of str): The input text or list of texts to chunk.

        Returns:
            list of str: A list of text chunks.
        """
        chunks = []
        if isinstance(data, str):
            data = [data]
        for text in data:
            for i in range(0, len(text), self.chunk_size):
                max_size = min(self.chunk_size, len(text) - i)
                chunks.append(clean_text(text[i:i + max_size]))
        return chunks


class OverlapChunkingStrategy:
    """Splits text into chunks of a specified size with an overlap between chunks.

    Attributes:
        chunk_size (int): The maximum size of each chunk.
        overlap_size (int): The number of characters each chunk overlaps with the next.
    """

    def __init__(self, chunk_size=512, overlap_size=32):
        """Initializes the OverlapChunkingStrategy with specified chunk and overlap sizes.

        Args:
            chunk_size (int): The maximum size of each chunk.
            overlap_size (int): The number of characters to overlap between chunks.
        """
        assert chunk_size > overlap_size, "Chunk size must be greater than overlap size"
        assert overlap_size > 0, "Overlap size must be greater than 0"
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def get_chunks(self, data):
        """Splits the input text into overlapping chunks based on chunk_size and overlap_size.

        Args:
            data (str or list of str): The input text or list of texts to chunk.

        Returns:
            list of str: A list of overlapping text chunks.
        """
        chunks = []
        if isinstance(data, str):
            data = [data]
        for text in data:
            for i in range(0, len(text), self.chunk_size - self.overlap_size):
                max_size = min(self.chunk_size, len(text) - i)
                chunks.append(clean_text(text[i:i + max_size]))
        return chunks


class NaturalBreakOverlapStrategy:
    """Chunks text into natural segments based on sentence end while respecting the max_chunk_size.

    This strategy splits text at natural sentence boundaries, aiming for chunks that do not exceed a specified maximum size. It's particularly useful for creating readable segments of text that are convenient for processing.

    Attributes:
        chunk_size (int): The maximum size of each text chunk.
    """

    def __init__(self, chunk_size=256):
        """Initializes the NaturalBreakOverlapStrategy with a specified chunk size.

        Args:
            chunk_size (int): The maximum size of each text chunk. Defaults to 256.
        """
        self.chunk_size = chunk_size

    def get_chunks(self, data):
        """Chunks the text into natural segments based on sentence end while respecting the max_chunk_size.

        Args:
            data (str or list of str): The input text or list of texts to chunk.

        Returns:
            list of str: A list of text chunks, each a natural segment based on sentence boundaries.
        """
        chunks = []
        current_chunk = []
        if isinstance(data, str):
            data = [data]
        for text in data:
            sentences = re.split(r'(?<=[.!?]) +', text)
            for sentence in sentences:
                sentence = clean_text(sentence) + ' '  # Clean and add space back for readability
                if len(' '.join(current_chunk) + sentence) > self.chunk_size:
                    chunks.append(' '.join(current_chunk).strip())
                    current_chunk = [sentence]
                else:
                    current_chunk.append(sentence)
            if current_chunk:
                chunks.append(' '.join(current_chunk).strip())
        return chunks


class NLPOverlapStrategy:
    """Chunks text into natural segments based on sentence end using spaCy, with overlap.

    This strategy utilizes spaCy's natural language processing (NLP) capabilities to identify sentence boundaries and chunk text accordingly. It allows for an optional overlap of sentences between consecutive chunks, enhancing the coherence of chunk boundaries in certain applications.

    Attributes:
        chunk_size (int): The maximum size of each chunk.
        num_of_sentences_to_overlap (int): The number of sentences to overlap between consecutive chunks.
        nlp (spacy.lang): Loaded spaCy language model for sentence boundary detection.
    """

    def __init__(self, chunk_size=256, num_of_sentences_to_overlap=1):
        """Initializes the NLPOverlapStrategy with specified chunk size and sentence overlap.

        Args:
            chunk_size (int): The maximum size of each chunk.
            num_of_sentences_to_overlap (int): The number of sentences to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.num_of_sentences_to_overlap = num_of_sentences_to_overlap
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy English language model

    def get_chunks(self, data):
        """Chunks the text into natural segments based on sentence end using spaCy, with overlap.

        Args:
            data (str or list of str): The input text or list of texts to chunk.

        Returns:
            list of str: A list of text chunks, each a natural segment based on sentence boundaries with optional overlap.
        """
        chunks = []
        current_chunk = []
        if isinstance(data, str):
            data = [data]
        for text in data:
            doc = self.nlp(text)
            sentences = [sentence.text.strip() for sentence in doc.sents]  # Extract sentences as clean text
            for sentence in sentences:
                current_chunk.append(clean_text(sentence))
                chunk_text = ' '.join(current_chunk).strip()
                if len(chunk_text) >= self.chunk_size:
                    chunks.append(chunk_text)
                    if 0 < self.num_of_sentences_to_overlap < len(current_chunk):
                        current_chunk = current_chunk[-self.num_of_sentences_to_overlap:]
                    else:
                        current_chunk = []
            if current_chunk:
                chunks.append(' '.join(current_chunk).strip())
        return chunks
