import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model='text-embedding-3-large', dimensions=32,
    openai_api_key=os.getenv("GITHUB_TOKEN"),
    openai_api_base="https://models.inference.ai.azure.com"
)
result = embedding.embed_query("Virat Kohli is the fastest to get 10000 runs in ODI cricket.");
print(str(result)  )
