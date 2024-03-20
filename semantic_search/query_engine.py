from openai import OpenAI
from utils import OPENAI_API_KEY, SYSTEM_PROMPT, ANTHROPIC_API_KEY
from anthropic import Anthropic

class OpenaiQueryEngine:
  """
  A query engine for interacting with OpenAI's GPT models.

  Attributes:
      data_index (object): The data index to query for related information.
      num_results (int): The number of results to retrieve.
      client (OpenAI): The OpenAI client configured with an API key.

  Parameters:
      data_index (object): The data index to be used for querying.
      num_results (int, optional): The default number of results to retrieve. Defaults to 10.
  """
  def __init__(self, data_index, num_results=10):
    self.data_index = data_index
    self.num_results = num_results
    self.client = OpenAI(api_key=OPENAI_API_KEY)

  def answer(self, query):
    """
    Generates an answer to a given query using the OpenAI API.

    Parameters:
        query (str): The user's query to answer.

    Returns:
        str: The generated answer to the query.
    """
    similar = self.data_index.query(query)
    response = self.client.chat.completions.create(
      model='gpt-4-0125-preview',
      messages=[
        {"role": "system", "content": f"{SYSTEM_PROMPT}."},
        {"role": "user", "content": f"{query}. Explicit knowledge base: {similar}"}
      ]
    )
    return response.choices[0].message.content.strip()


class AnthropicQueryEngine:
  """
  A query engine for interacting with Anthropic's Claude models.

  Attributes:
      data_index (object): The data index to query for related information.
      num_results (int): The number of results to retrieve.
      client (Anthropic): The Anthropic client configured with an API key.

  Parameters:
      data_index (object): The data index to be used for querying.
      num_results (int, optional): The default number of results to retrieve. Defaults to 10.
  """
  def __init__(self, data_index, num_results=10):
    self.data_index = data_index
    self.num_results = num_results
    self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

  def answer(self, query):
    """
    Generates an answer to a given query using the Anthropic API.

    Parameters:
        query (str): The user's query to answer.

    Returns:
        str: The generated answer to the query.
    """
    similar = self.data_index.query(query)
    response = self.client.messages.create(
      model='claude-3-opus-20240229',
      temperature=0,
      max_tokens=1024,
      system=SYSTEM_PROMPT,
      messages=[
        {
          "role": "user",
          "content": [{
            "text": f"{query}. Explicit knowledge base: {similar}",
            "type": "text"
          }]
        }
      ]
    )
    return response.content[0].text.strip()
