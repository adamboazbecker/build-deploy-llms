import os
import pathlib
from typing import List, Tuple
import openai
import tiktoken
from getpass import getpass
import wandb

import langchain
from langchain.docstore.document import Document
from langchain import document_loaders
from config import config

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


QUERY = "How should I think about artifacts?"


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


def load_documents(data_dir: str) -> List[Document]:
  """Load documents from a directory of markdown files

  Args:
      data_dir (str): The directory containing the markdown files

  Returns:
      List[Document]: A list of documents
  """
  md_files = list(map(str, pathlib.Path(data_dir).glob("*.md")))
  documents = [
    document_loaders.UnstructuredMarkdownLoader(file_path=file_path).load()[0]
    for file_path in md_files
  ]
  return documents


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
  embedding_function = langchain.embeddings.OpenAIEmbeddings(openai_api_key=openai.api_key)
  vector_store = langchain.vectorstores.Chroma.from_documents(
    documents=documents,
    embedding=embedding_function,
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
  retriever = vector_store.as_retriever(search_kwargs=dict(k=3))
  docs = retriever.get_relevant_documents(query)
  # Let's see the results
  for doc in docs:
    print(doc.metadata["source"])


def log_index(vector_store_dir: str, run: "wandb.run"):
  """Log a vector store to wandb

  Args:
      vector_store_dir (str): The directory containing the vector store to log
      run (wandb.run): The wandb run to log the artifact to.
  """
  index_artifact = wandb.Artifact(name="vector_store", type="search_index")
  index_artifact.add_dir(vector_store_dir)
  run.log_artifact(index_artifact)


def main():
  # Step 1:
  run = wandb.init(project=config['project'], config=config)

  # Step 2:
  read_openai_api_key()

  # Step 3:
  documents = load_documents("./docs_sample")

  # Step 4:
  vector_store = create_vector_store(documents, vector_store_path="./vector_store")

  # Step 5:
  get_relevant_documents(QUERY, vector_store)

  # Step 6:
  log_index("./docs_sample", run)

  # Step 7:
  run.finish()


if __name__ == "__main__":
  main()

