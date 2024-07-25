import chromadb
client = chromadb.PersistentClient()
collection_name = 'bukhari_muslim_collection'
collection = client.get_collection(collection_name)

# Giving hadith in query to get 10 similar ahadith from chromadb collection
results = collection.query(
    query_texts=['''Narrated `Imran:                     We performed Hajj-at-Tamattu` in the lifetime of Allah's Apostle and then the Qur'an was revealed (regarding Hajj-at-Tamattu`) and somebody said what he wished (regarding Hajj-at-Tamattu`) according his own opinion.'''],
    n_results=10
)
print(results)