from chunking import NLPOverlapStrategy
from data_index import InMemoryIndex, PineconeIndex
from data_loader import DataLoader
from embedding import OpenSourceEmbedding
from query_engine import OpenaiQueryEngine, AnthropicQueryEngine


class RetrievalAugmentedRunner:
    """
    A class to execute retrieval-augmented operations using a specified query engine and data indexer.

    Attributes:
        k (int): The number of top results to retrieve.
        index: The data indexer instance used for indexing and retrieval.
        query: The query engine instance used for generating answers based on the indexed data.
    """

    def __init__(self, query_engine, data_indexer, k=5):
        """
        Initializes the RetrievalAugmentedRunner with a query engine, data indexer, and optional k value.

        Parameters:
            query_engine: An instance of a query engine (e.g., OpenaiQueryEngine or AnthropicQueryEngine).
            data_indexer: An instance of a data indexer (e.g., InMemoryIndex or PineconeIndex).
            k (int, optional): The number of results to retrieve. Defaults to 5.
        """
        self.k = k
        self.index = data_indexer
        self.query = query_engine

    def __call__(self, query):
        """
        Executes the query using the query engine and returns the top k results.

        Parameters:
            query (str): The user query to process.

        Returns:
            The top k results from the query engine based on the indexed data.
        """
        return self.query.answer(query)


def main():
    """
    The main function to initialize components and run the retrieval-augmented system.
    """
    # Initialize the chunker with a specific chunk size.
    chunker = NLPOverlapStrategy(chunk_size=256)
    # Initialize the data loader with a specified directory, batch size, and chunking strategy.
    data_loader = DataLoader('data', batch_size=128, chunker=chunker)

    # Initialize the data index with the data loader and embedding method.
    # This can be either an in-memory index or a Pinecone vector database index.
    index = InMemoryIndex(data_loader, embed=OpenSourceEmbedding())
    # index = PineconeIndex(data_loader, embed=OpenSourceEmbedding())

    # Initialize the query engine with the index and a specific number of results to return.
    query_engine = OpenaiQueryEngine(index, 3)
    # query_engine = AnthropicQueryEngine(index, 3)

    # Initialize the runner with the query engine, index, and k value.
    runner = RetrievalAugmentedRunner(query_engine, index, k=3)

    print("Suggested questions for the sample data file are:\n")
    print("Comment on the labour force within the home care sector")
    print("What changes have been made to the labour force in the home care sector?")
    print("How are providers in the home care sector coping with challenges?")
    print("What are the challenges faced by providers in the home care sector?")
    print("-----------------------------------------------------------------------------------\n\n")
    while True:
        # Input loop for processing user queries.
        question = input("Enter your question(:q to quit): ")
        if question == ":q":
            break
        result = runner(question)
        print(result)
        print("\n=====================================================\n")


if __name__ == "__main__":
    main()
