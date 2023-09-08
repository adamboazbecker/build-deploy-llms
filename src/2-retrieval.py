import os
import pathlib
from typing import List
import openai
import tiktoken
from getpass import getpass

import langchain
from langchain.docstore.document import Document
from langchain import document_loaders
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

QUERY = "Hi there!"


def read_openai_api_key():
  """
  Reads the OpenAI API key from environment variables or user input.

  This function first attempts to retrieve the OpenAI API key from the environment variable
  "OPENAI_API_KEY." If the environment variable is not set, it prompts the user to input
  their API key interactively. The API key is then validated to ensure it starts with "sk-"
  to confirm its validity, and it is set as the authentication key for OpenAI API calls.

  Args:
      None

  Raises:
      AssertionError: If the provided API key does not start with "sk-", indicating an
                       invalid OpenAI API key.

  Returns:
      None
  """
  api_key = os.environ.get("OPENAI_API_KEY", None)
  if api_key is None:
    api_key = getpass("Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")

  assert api_key.startswith("sk-"), "This doesn't look like a valid OpenAI API key"
  openai.api_key = api_key



def play_with_embeddings():
  """
  Demonstrates token encoding and decoding using the tiktoken library.

  This function encodes the input text "NYC is the place to be" into tokens using tiktoken's
  encoding_for_model method, prints the encoded tokens, and then decodes them one by one.
  The purpose of this function is to illustrate how token encoding and decoding works.

  Args:
      None

  Prints:
      Encoded tokens for the input text.
      Decoded tokens one by one, along with their corresponding token IDs.
  """
  enc = tiktoken.encoding_for_model("NYC is the place to be")
  print(enc)
  print(tiktoken.encoding.decode(enc))

  # we can decode the tokens one by one
  for token_id in enc:
    print(f"{token_id}\t{tiktoken.encoding.decode([token_id])}")


def load_documents(data_dir: str) -> List[Document]:
  """Load documents from a directory of markdown files

  Args:
      data_dir (str): The directory containing the markdown files

  Returns:
      List[Document]: A list of documents
  """
  md_files = list(map(str, pathlib.Path(data_dir).glob("*.md")))
  documents = [
    # Clue! Load using the UnstructuredMarkdownLoader imported above
    # [Enter the loading function]
    for file_path in md_files
  ]

  """
  Each call to `load` returns the following response:
  [
    Document(page_content='the text in the document', metadata={'source': 'the/file/path.md'})
  ]
  """
  return documents


def chunk_documents():
  """
  Placeholder function for document chunking.

  Extra challenge: This function is intended to perform the task of dividing large documents or text into smaller,
  manageable chunks or segments. The specific implementation details for document chunking
  are not provided in this placeholder function.

  Args:
      None

  Returns:
      None
  """
  pass


def create_vector_store(
  documents,
  vector_store_path: str = "./vector_store",
) -> langchain.vectorstores.Chroma:
  """Create a ChromaDB vector store from a list of documents

  Args:
      documents (_type_): A list of documents to add to the vector store
      vector_store_path (str, optional): The path to the vector store. Defaults to "./vector_store".

  Returns:
      Chroma: A ChromaDB vector store containing the documents.
  """

  #
  # Clue! We need to use an embedding in order to create a vector store
  #
  vector_store = langchain.vectorstores.Chroma.from_documents(
    documents=documents,
    persist_directory=vector_store_path,
  )
  vector_store.persist()
  return vector_store


def get_relevant_documents(query, vector_store):
  """
  Retrieves relevant documents from a vector store based on a given query.

  Args:
      query (str): The query string to search for relevant documents.
      vector_store (VectorStore): An instance of the VectorStore class containing document vectors.

  Returns:
      list: A list of relevant documents retrieved from the vector store.

  Prints:
      Prints the source metadata of each relevant document to the console.
  """
  docs = vector_store.get_relevant_documents(query)
  # Let's see the results
  for doc in docs:
    print(doc.metadata["source"])


def main():
  # Step 1:
  read_openai_api_key()

  # Step 2 (optional):
  #   Build up some intuition about the embeddings with the following function: play_with_embeddings.

  # Step 3:
  documents = load_documents("./docs_sample")

  # Step 4:
  vector_store = create_vector_store(documents, vector_store_path="./vector_store")

  # Step 5:
  get_relevant_documents(QUERY, vector_store)

  # Step 6 (optional):
  #   Extra challenge: If you finish early, you should try to chunk the documents into smaller chunks!
if __name__ == "__main__":
  main()
