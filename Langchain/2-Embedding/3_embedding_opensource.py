import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
result = embedding.embed_query("Virat Kohli is the fastest to get 10000 runs in ODI cricket.")
print(result)    