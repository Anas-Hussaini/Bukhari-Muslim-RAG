#### RETRIEVAL and GENERATION ####
import os
from dotenv import load_dotenv
import chromadb
import openai

load_dotenv(
    dotenv_path="D:/Data Science Projects/Scalence Internship ML/Bukhari-Muslim-RAG/.env",
    override=True
)

OpenAI_TOKEN = os.environ["OpenAI_TOKEN"]
print(OpenAI_TOKEN)

import chromadb

collection_name = 'sherlock_holmes_collection'

import chromadb.utils.embedding_functions as embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OpenAI_TOKEN,
                model_name="text-embedding-ada-002"
            )

vectorstore = chromadb.PersistentClient()
collection = vectorstore.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef
)

question = input("Enter the question:")

retriever = collection.query(
    query_texts=question,
    n_results=3
)

print(retriever)

# retriever

type(retriever)
def format_docs(docs):
    return "\n\n".join(docs['documents'][0])

context = format_docs(retriever)

base_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {} 

Context: {} 

Answer:
"""

prompt = f'{base_prompt.format(question, context)}'
prompt

# def RunnablePassthrough(input_question):
#     # This function simply returns the input data without modification
#     return input_question

from openai import OpenAI

openai_client = OpenAI(
    api_key=OpenAI_TOKEN
)
 
answer = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    temperature=0,
    messages=[
        {"role": "system", "content": prompt}
    ]
)

print(answer.choices[0].message.content)








