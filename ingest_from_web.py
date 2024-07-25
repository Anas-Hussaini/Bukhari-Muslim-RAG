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

# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
import bs4

# loader = TextLoader("story_sherlock_holmes.txt")
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()
docs

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

from langchain_community.vectorstores import Chroma
import chromadb
from langchain_openai import OpenAIEmbeddings

collection_name = 'sherlock_holmes_collection'
persist_directory = "D:/Data Science Projects/Scalence Internship ML/Bukhari-Muslim-RAG/chroma"
path="D:/Data Science Projects/Scalence Internship ML/Bukhari-Muslim-RAG/chroma"

# vectorstore = Chroma("langchain_store", embeddings)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding = OpenAIEmbeddings(api_key=OpenAI_TOKEN),
    collection_name = collection_name,
    persist_directory = persist_directory,
    client=chromadb.PersistentClient(path=path)
)