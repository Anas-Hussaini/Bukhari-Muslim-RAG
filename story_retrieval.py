#### RETRIEVAL and GENERATION ####
import os
from dotenv import load_dotenv, dotenv_values

load_dotenv(
    dotenv_path="D:/Data Science Projects/Scalence Internship ML/Bukhari-Muslim-RAG/.env",
    override=True
)

LS_TOKEN = os.environ["LS_TOKEN"]
print(LS_TOKEN)
OpenAI_TOKEN = os.environ["OpenAI_TOKEN"]
print(OpenAI_TOKEN)

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = LS_TOKEN

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


collection_name = 'sherlock_holmes_collection'

# chroma_instance = Chroma()

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(api_key=OpenAI_TOKEN),
    collection_name=collection_name
)

retriever = vectorstore.as_retriever()

# set the LANGCHAIN_API_KEY environment variable (create key in settings)
from langchain import hub
# prompt = hub.pull("story-question-answer/rag-story-question")

# Prompt
prompt = hub.pull("rlm/rag-prompt")
prompt
# prompt = hub.pull("rag-story-question")

# prompt

from langchain_openai import ChatOpenAI

# LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    api_key=OpenAI_TOKEN
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# from langchain_openai import OpenAIEmbeddings

# Question
# story_sherlock_holmes.txt
# Who is James McCarthy?
question = input("Please enter the question: ")
answer = rag_chain.invoke(question)
print(answer)