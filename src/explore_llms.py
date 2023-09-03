import os
import openai
from getpass import getpass
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


def read_openai_api_key():
  api_key = os.environ.get("OPENAI_API_KEY", None)
  if api_key is None:
    api_key = getpass("Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")

  assert api_key.startswith("sk-"), "This doesn't look like a valid OpenAI API key"
  openai.api_key = api_key

def play_with_temperature():
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Say something about New York City",
    max_tokens=100,
    temperature=0.02,
  )


def play_with_topp():
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Say something about New York City",
    max_tokens=100,
    top_p=1
  )


def play_with_chat():
  MODEL = "gpt-3.5-turbo"
  response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say something about New York City"},
    ],
    temperature=0,
  )


if __name__ == "__main__":
  read_openai_api_key()
  play_with_temperature()
  play_with_topp()
  play_with_chat()
