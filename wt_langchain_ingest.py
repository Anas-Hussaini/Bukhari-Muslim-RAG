import os
from dotenv import load_dotenv, dotenv_values
import openai

load_dotenv(
    dotenv_path="D:/Data Science Projects/Scalence Internship ML/Bukhari-Muslim-RAG/.env",
    override=True
)

OpenAI_TOKEN = os.environ["OpenAI_TOKEN"]
print(OpenAI_TOKEN)

# url = input("Please enter the path of file to ingest: ")
url="story_sherlock_holmes.txt"

with open(url, 'r', encoding='utf-8') as file:
    docs = file.read()

print(docs)

import re
from typing import List

class RecursiveCharacterTextSplitter:
    def __init__(self, max_length: int, delimiters: List[str] = None):
        self.max_length = max_length
        self.delimiters = delimiters or ['\n\n', '\n', '.', '!', '?', ' ', '']

    def split_text(self, text: str) -> List[str]:
        return self._recursive_split(text, 0)

    def _recursive_split(self, text: str, delimiter_index: int) -> List[str]:
        if len(text) <= self.max_length:
            return [text]

        if delimiter_index >= len(self.delimiters):
            return [text[i:i+self.max_length] for i in range(0, len(text), self.max_length)]

        delimiter = self.delimiters[delimiter_index]
        if delimiter:
            parts = text.split(delimiter)
        else:
            parts = list(text)  # Split into individual characters if delimiter is empty

        splits = []
        current_split = ""

        for part in parts:
            if current_split and len(current_split) + len(part) + len(delimiter) > self.max_length:
                splits.append(current_split)
                current_split = part + delimiter
            else:
                current_split += part + delimiter

        if current_split:
            splits.append(current_split)

        # Further split splits that are too large
        final_splits = []
        for split in splits:
            if len(split) > self.max_length:
                final_splits.extend(self._recursive_split(split, delimiter_index + 1))
            else:
                final_splits.append(split)

        return final_splits

splitter = RecursiveCharacterTextSplitter(max_length=50)
splits = splitter.split_text(docs)
print(splits)
len(splits)

import chromadb

path="D:/Data Science Projects/Scalence Internship ML/Bukhari-Muslim-RAG/chroma"

vectorstore = chromadb.PersistentClient(
    path=path
)

collection_name = 'sherlock_holmes_collection_without_langchain'

import chromadb.utils.embedding_functions as embedding_functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OpenAI_TOKEN,
                model_name="text-embedding-ada-002"
            )

collection = vectorstore.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef
)

len(splits)
ids = []
len(ids)
id = 1
for i, split in enumerate(splits):
        ids.append(str(id))
        id+=1
# ids
len(ids)

import math

batch_size = 2000
loops = math.ceil((len(splits))/batch_size)
loops

for i in range(0,loops):
    collection.upsert(
        documents = splits[(i*batch_size):(batch_size*(i+1))],
        ids=ids[i*batch_size:batch_size*(i+1)]
        # embedding = OpenAIEmbeddings(api_key=OpenAI_TOKEN)
        # embeddings = openai.embeddings.create()
    )
