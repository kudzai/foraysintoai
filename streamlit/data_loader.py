from chunking_strategy import RecursiveSplitterChunkingStrategy
from parse_file import parse


class DataLoader:
    def __init__(self, chunking_strategy=None, batch_size=512):
        self.chunking_strategy = chunking_strategy if chunking_strategy is not None else RecursiveSplitterChunkingStrategy()
        self.batch_size = batch_size

    def get_chunks(self, file):
        """
        Uses the chunker strategy to get chunks from the loaded file.

        Since 'parse' yields parsed content or None, this method needs to handle the generator.

        Returns:
            A generator yielding chunks of data as defined by the chunking strategy.
        """
        parsed_content_generator = parse(file)
        for content in parsed_content_generator:
            if content is not None:  # Ensure content is not None before proceeding
                yield from self.chunking_strategy.get_chunks(content)
            else:
                # Handle unsupported file type or error in parsing
                print("Error or unsupported file type. No chunks to yield.")
                yield from []  # Yield nothing

    def get_chunk_batches(self, file):
        """
        Yields batches of data chunks, with each batch containing up to the specified batch size.

        If the final batch has fewer chunks than the batch size, it will still be yielded.

        Yields:
            A list of chunks, where each list is a batch of data.
        """
        chunks = []
        for chunk in self.get_chunks(file):
            chunks.append(chunk)
            if len(chunks) == self.batch_size:
                yield chunks
                chunks = []
        if chunks:  # Yield any remaining chunks as the final batch
            yield chunks
