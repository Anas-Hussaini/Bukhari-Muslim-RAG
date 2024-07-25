# Import sentence transformer library and choose model to make embeddings of ahadith
from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer('all-MiniLM-L6-v2')

# Import csv library
import csv

# Load saved csv file to make a format of data to be stored in Chromadb collection
with open('D:/Data Science Projects/Scalence Internship ML/Bukhari-Muslim-RAG/first_50.csv') as file:
    lines = csv.reader(file)
    
    documents_b_m = []
    embeddings_b_m = []
    metadatas_b_m = []
    ids_b_m = []
    id = 1
    
    for i, line in enumerate(lines):
        if i==0:
            continue
        
        #Code to store ahadith in variables with embeddings, metadatas and ids
        documents_b_m.append(line[2])
        embeddings_b_m.append(model.encode(line[2]).tolist())
        metadatas_b_m.append({"hadith_id": line[1],"source": line[0]})
        ids_b_m.append(str(id))
        id+=1
        
# Import Chromadb library and config as default settings
import chromadb

# Make a client on Chromadb
client = chromadb.PersistentClient()

# Naming a collection to be made
collection_name = 'bukhari_muslim_collection'

# When creating a collection
collection = client.get_or_create_collection(collection_name)

# Import math library
import math

# Creating loop variable for "for loop"
batch_size=5461
loops = math.ceil((len(documents_b_m))/batch_size)
loops

# Storing data from variables to chromadb collection
for i in range(0,loops):
    collection.upsert(
        documents=documents_b_m[(i*batch_size):(batch_size*(i+1))],
        metadatas=metadatas_b_m[i*batch_size:batch_size*(i+1)],
        ids=ids_b_m[i*batch_size:batch_size*(i+1)],
        embeddings=embeddings_b_m[i*batch_size:batch_size*(i+1)]
    )