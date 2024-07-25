import openai
from dotenv import load_dotenv

load_dotenv(
    dotenv_path="D:/Data Science Projects/Scalence Internship ML/Bukhari-Muslim-RAG/.env",
    override=True
)

OpenAI_TOKEN = os.environ["OpenAI_TOKEN"]

openai.api_key = OpenAI_TOKEN

texts = ["This is the first string.",
         "Here is another string.",
         "And yet another string for embeddings."]

data = []

for text in texts:
    response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    data.append(response)
    
data
type(data)

embeddings = []
embedding_data = []

for response in data:
    embedding_data = response["data"]
    for item in embedding_data:
        embeddings.append(item["embedding"])
        




