import openai
from dotenv import load_dotenv

load_dotenv(
    dotenv_path="D:/Data Science Projects/Scalence Internship ML/Bukhari-Muslim-RAG/.env",
    override=True
)

OpenAI_TOKEN = os.environ["OpenAI_TOKEN"]

openai.api_key = OpenAI_TOKEN

# List of strings you want to embed
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial Intelligence is transforming the world.",
    "OpenAI provides advanced AI tools and models."
]

# Generate embeddings for each string
response = openai.embeddings.create(
    input=texts,
    model="text-embedding-ada-002"  # Choose the model that fits your needs
)
response
# Extract embeddings from the response
embeddings = [item['embedding'] for item in response['data']]

for text, embedding in zip(texts, embeddings):
    print(f"Text: {text}\nEmbedding: {embedding}\n")

