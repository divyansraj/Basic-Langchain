import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

embedding = OpenAIEmbeddings(
    model='text-embedding-3-large', dimensions=32,
    openai_api_key=os.getenv("GITHUB_TOKEN"),
    openai_api_base="https://models.inference.ai.azure.com"
)
documents = [
"Sir Donald Bradman (Australia) is a cricketing legend known for holding an astonishing and unbroken Test batting average of 99.94.",
"Sachin Tendulkar (India) is often called the \"Master Blaster\" and holds the unique record of scoring 100 international centuries in his career.",
"Virat Kohli (India) is a modern-day superstar and highly influential figure in the sport, celebrated for his aggressive batting and remarkable consistency across all formats.",
"Sir Garfield Sobers (West Indies) is widely regarded as history's greatest all-rounder due to his exceptional skills in batting, fast bowling, and spin bowling."
]

embedding_documents = embedding.embed_documents(documents);

query = "Who is modern superstar in cricket?"

query_embedding = embedding.embed_query(query);

scores = cosine_similarity([query_embedding], embedding_documents)[0];

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1];

print("Result for query:", query );
print("Most similar document:", documents[index] );
print("Similarity score:", score );