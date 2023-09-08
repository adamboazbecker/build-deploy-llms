"""This module contains functions for loading a ConversationalRetrievalChain"""

import logging
import wandb
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


logger = logging.getLogger(__name__)


def load_vector_store(wandb_run: wandb.run, openai_api_key: str) -> Chroma:
    """Load a vector store from a Weights & Biases artifact
    Args:
        run (wandb.run): An active Weights & Biases run
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        Chroma: A chroma vector store object
    """
    # load vector store artifact
    vector_store_artifact_dir = wandb_run.use_artifact(
        wandb_run.config['vector_store_artifact'], type="search_index"
    ).download()
    embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # load vector store
    vector_store = Chroma(
        embedding_function=embedding_fn, persist_directory=vector_store_artifact_dir
    )

    return vector_store


def load_chain(wandb_run: wandb.run, vector_store: Chroma, openai_api_key: str):
    """Load a ConversationalQA chain from a config and a vector store
    Args:
        wandb_run (wandb.run): An active Weights & Biases run
        vector_store (Chroma): A Chroma vector store object
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        ConversationalRetrievalChain: A ConversationalRetrievalChain object
    """
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=wandb_run.config['model_name'],
        temperature=wandb_run.config['chat_temperature'],
        max_retries=wandb_run.config['max_fallback_retries'],
    )
    qa_prompt = load_chat_prompt()
    qa_chain = ConversationalRetrievalChain(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )
    return qa_chain


def load_chat_prompt() -> ChatPromptTemplate:
    """
     Loads a chat prompt template for Wandbot, an AI assistant for answering Weights & Biases (WandB) related questions.

     This function defines a chat prompt template consisting of a system template that instructs
     the AI assistant (Wandbot) on how to respond to user queries related to Weights & Biases (WandB).
     The template includes guidelines on providing accurate and helpful responses based on context
     and not relying on prior knowledge.

     Args:
         None

     Returns:
         ChatPromptTemplate: A template for conducting a conversation with Wandbot.

     Note:
         - The template defines Wandbot's behavior, including handling unrelated questions and
           asking for clarification when necessary.
         - Wandbot's responses are generated based on the context provided in the conversation.
     """
    template = {
        "system_template": "You are wandbot, an AI assistant designed to provide accurate and helpful responses to"
           "questions related to Weights & Biases and its python SDK, wandb.\\n"
           "Your goal is to always provide conversational answers based solely on the context"
           "information provided by the user and not rely on prior knowledge.\\"
           "nWhen possible, provide code blocks and HTTP links directly from the official"
           "documentation at https://docs.wandb.ai, but ensure that they are relevant and not"
           "fabricated.\\n\\n"
           "If you are unable to answer a question or generate valid code or links based on the"
           "context provided, respond with 'Hmm, I'm not sure' and direct the user to post the"
           "question on the community forums at https://community.wandb.ai/ or reach out to"
           "wandb support via support@wandb.ai.\\n\\n"
           "You can only answer questions related to wandb and Weights & Biases."
           "\\nIf a question is not related, politely inform the user and offer to assist with"
           "any wandb-related questions they may have.\\n\\"
           "nIf necessary, ask follow-up questions to clarify the context and provide a more"
           "accurate answer.\\n\\nThank the user for their question and offer additional"
           "assistance if needed.\\nALWAYS prioritize accuracy and helpfulness in your responses."
           "\\n\\nHere is an example conversation:\\n\\n"
           "CONTEXT"
           "\\nContent: Weights & Biases supports logging audio data arrays or file that can be"
           "played back in W&B. You can log audio with `wandb.Audio()`"
           "\\n\\nContent: # Log an audio array or file\\n"
           "wandb.log({{'my whale song': wandb.Audio(\\n    array_or_path, caption='montery whale 0034',"
           " sample_rate=32)}})\\n\\n# "
           "OR\\n\\n# Log your audio as part of a W&B "
           "Table\\nmy_table = wandb.Table(columns=['audio', 'spectrogram', 'bird_class', 'prediction'])"
           "\\nfor (audio_arr, spec, label) in my_data:\\n       "
           "pred = model(audio)\\n\\n      "
           " # Add the data to a W&B Table\\n       audio = wandb.Audio(audio_arr, sample_rate=32)\\n       "
           "img = wandb.Image(spec)\\n       my_table.add_data(audio, img, label, pred)\\n\\n"
           "# Log the Table to wandb\\n wandb.log({{'validation_samples' : my_table}})'"
           "\\n\\n================\\nQuestion: Hi, @wandbot: How can I log audio with wandb?"
           "\\n================\\nFinal Answer in Markdown: Here is an example of how to log audio with "
           "wandb:\\n\\n```\\nimport wandb\\n\\n# Create an instance of the wandb.data_types.Audio "
           "class\\naudio = wandb.data_types.Audio(data_or_path='path/to/audio.wav', sample_rate=44100, "
           "caption='My audio clip')\\n\\n# Get information about the audio clip\\ndurations = "
           "audio.durations()\\nsample_rates = audio.sample_rates()\\n\\n# Log the audio clip"
           "\nwandb.log({{'audio': audio}})\\n```\\n\\n\\nCONTEXT\\n================\\nContent: "
           "ExtensionArray.repeat(repeats, axis=None) Returns a new ExtensionArray where each element "
           "of the current ExtensionArray is repeated consecutively a given number of times."
           "\\n\\nParameters: repeats int or array of ints. The number of repetitions for each element. "
           "This should be a positive integer. Repeating 0 times will return an empty array. "
           "axis (0 or \\u2018index\\u2019, 1 or \\u2018columns\\u2019), default 0 The axis along "
           "which to repeat values. Currently only axis=0 is supported.\\n\\n================\\n"
           "Question: How to eat vegetables using pandas?\\n================\\nFinal Answer in Markdown:"
           " Hmm, The question does not seem to be related to wandb. As a documentation bot for wandb "
           "I can only answer questions related to wandb. Please try again with a question related "
           "to wandb.\\n\\n\\nBEGIN\\n================\\nCONTEXT\\n{context}"
           "\\n================\\nGiven the context information and not prior knowledge, answer "
           "the question.\\n================\\n",
        "human_template": "{question}\\n================\\nFinal Answer in Markdown:"
    }

    messages = [
        SystemMessagePromptTemplate.from_template(template["system_template"]),
        HumanMessagePromptTemplate.from_template(template["human_template"]),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt


def get_answer(
    chain: ConversationalRetrievalChain,
    question: str,
    chat_history: list[tuple[str, str]],
):
    """Get an answer from a ConversationalRetrievalChain
    Args:
        chain (ConversationalRetrievalChain): A ConversationalRetrievalChain object
        question (str): The question to ask
        chat_history (list[tuple[str, str]]): A list of tuples of (question, answer)
    Returns:
        str: The answer to the question
    """
    result = chain(
        inputs={"question": question, "chat_history": chat_history},
        return_only_outputs=True,
    )
    response = f"Answer:\t{result['answer']}"
    return response

