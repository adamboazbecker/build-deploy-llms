import os
import openai
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from getpass import getpass

import langchain
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


def read_openai_api_key():
  api_key = os.environ.get("OPENAI_API_KEY", None)
  if api_key is None:
    api_key = getpass("Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")

  assert api_key.startswith("sk-"), "This doesn't look like a valid OpenAI API key"
  openai.api_key = api_key


def download_from_wandb_artifact():
  # Implement wandb init here
  return "./vector_store"


def load_vector_store(vector_store_path) -> langchain.vectorstores.Chroma:
  """Load a vector store from a Weights & Biases artifact
  Args:
      vector_store_path (str): The path to the vector store
  Returns:
      Chroma: A chroma vector store object
  """
  embedding_fn = langchain.embeddings.OpenAIEmbeddings(openai_api_key=openai.api_key)
  # load vector store
  vector_store = langchain.vectorstores.Chroma(
    embedding_function=embedding_fn, persist_directory=vector_store_path
  )
  return vector_store


def get_relevant_documents(query, vector_store):
  retriever = vector_store.as_retriever(search_kwargs=dict(k=1))
  docs = retriever.get_relevant_documents(query)
  return docs


def get_stuffed_prompt(docs, query):
  prompt_template = """Use the following pieces of context to answer the question at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.

  {context}

  Question: {question}
  Helpful Answer:"""
  PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
  )

  context = "\n\n".join([doc.page_content for doc in docs])
  prompt = PROMPT.format(context=context, question=query)
  return prompt


def call_openai_chat(prompt):
  llm = OpenAI()
  response = llm.predict(prompt)


def set_up_logging():
  # What if we want to log and see everything in WandB afterwards?
  # we need a single line of code to start tracing langchain with W&B
  os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

  # wandb documentation to configure wandb using env variables
  # https://docs.wandb.ai/guides/track/advanced/environment-variables
  # here we are configuring the wandb project name
  os.environ["WANDB_PROJECT"] = "llmapps"


def retrieve_with_chain(question, vector_store):
  retriever = vector_store.as_retriever(search_kwargs=dict(k=1))

  qa = langchain.chains.RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever
  )
  result = qa.run(question)
  print(result)


def main():
  question = ""

  read_openai_api_key()
  vector_store_path = download_from_wandb_artifact()
  vector_store = load_vector_store(vector_store_path=vector_store_path)
  # Approach 1: "Stuff" the prompt yourself
  # documents = get_relevant_documents(question, vector_store)
  # stuffed_prompt = get_stuffed_prompt(documents, question)
  # call_openai_chat(stuffed_prompt)

  # Approach 2: Use a chain instead
  # set_up_logging()
  retrieve_with_chain(question, vector_store)


if __name__ == "__main__":
  main()
