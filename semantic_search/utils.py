from dotenv import load_dotenv
import os
import fitz  # PyMuPDF, a Python library to work with PDF files

# Load environment variables from a .env file located in the same directory as the script.
# This is useful for keeping sensitive data (like API keys) out of the source code.
load_dotenv()

# Retrieve specific environment variables (API keys) and store them in Python variables.
# This approach keeps sensitive keys secure and not hard-coded in the script.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def parse_pdf(file_path):
    """Parses the PDF at the given file_path and returns the extracted text.

    Args:
        file_path (str): The path to the PDF file to be parsed.

    Returns:
        str: The text extracted from the PDF.

    This function uses the PyMuPDF library to open and read through each page of the PDF,
    extracting all text. It's designed to handle any PDF file for text extraction purposes.
    """
    text = ''
    try:
        # Open the PDF file using PyMuPDF.
        with fitz.open(file_path) as doc:
            # Iterate through each page of the PDF.
            for page in doc:
                # Extract text from the current page and append it to the 'text' variable.
                text += page.get_text()
    except Exception as e:
        # If an error occurs during the PDF parsing, print the error message.
        print(f"Failed to parse PDF: {e}")
    return text


def read_text_file(file_path):
    """Reads the text file at the given file_path and returns the text.

    Args:
        file_path (str): The path to the text file to be read.

    Returns:
        str: The content of the text file.

    This function opens a text file, reads its content, and returns the text.
    It's a simple way to read text files for further processing or analysis.
    """
    with open(file_path, "r") as f:
        text = f.read()
    return text


# Defines a system prompt as a multi-line string. This is used as a guide or instruction
# for how an assistant or system should behave, specifically indicating that it should not
# use general knowledge but rather stick to an explicit knowledge base.
SYSTEM_PROMPT = '''
You are a helpful assistant that can only reference material from a knowledge base.
You do not like using any of your general knowledge.
You may only use information prefixed by "Explicit knowledge base:"
If the question cannot be answered only using information from the knowledge base, say instead "I'm sorry I cannot answer that."
Start every answer with a justification of whether their question can be answered from the explicit knowledge base provided.
'''.strip()
