"""A Simple chatbot that uses the LangChain and Gradio UI to answer questions about wandb documentation."""
import os
import gradio as gr
import wandb
import openai
from chain_utils import get_answer, load_chain, load_vector_store
from getpass import getpass
from dotenv import load_dotenv
from config import config

load_dotenv()  # take environment variables from .env.


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


class Chat:
    """A chatbot interface that persists the vectorstore and chain between calls."""

    def __init__(
        self,
        openai_key: str
    ):
        """Initialize the chatbot."""
        self.openai_key = openai_key
        self.config = config
        self.wandb_run = wandb.init(
            project=config['project'],
            entity=config['entity'],
            job_type=config['job_type'],
            config=config
        )
        self.vector_store = None
        self.chain = None

    def __call__(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None
    ):
        """Answer a question about wandb documentation using the LangChain QA chain and vector store retriever.
        Args:
            question (str): The question to answer.
            history (list[tuple[str, str]] | None, optional): The chat history. Defaults to None.
        Returns:
            list[tuple[str, str]], list[tuple[str, str]]: The chat history before and after the question is answered.
        """
        if self.vector_store is None:
            self.vector_store = load_vector_store(
                wandb_run=self.wandb_run, openai_api_key=self.openai_key
            )
        if self.chain is None:
            self.chain = load_chain(
                self.wandb_run, self.vector_store, openai_api_key=self.openai_key
            )

        history = history or []
        question = question.lower()
        response = get_answer(
            chain=self.chain,
            question=question,
            chat_history=history,
        )
        history.append((question, response))
        return history, history


with gr.Blocks() as demo:
    gr.HTML(
        """<div style="text-align: center; max-width: 700px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
            Build and Deploy LLM Apps Workshop
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Hi, I'm a helpful bot for answering questions. <br>
        To start, type your OpenAI API key, submit questions/issues you have related to your usage and then press enter.<br>
        </p>
    </div>"""
    )
    question = gr.Textbox(
        label="Type in your questions here and press Enter!",
        placeholder="How do I think about artifacts?",
    )
    state = gr.State()
    chatbot = gr.Chatbot()
    question.submit(
        Chat(openai_key=openai.api_key),
        [question, state],
        [chatbot, state],
    )


if __name__ == "__main__":
    demo.queue().launch(
        share=False, server_name="0.0.0.0", server_port=8884, show_error=True
    )
