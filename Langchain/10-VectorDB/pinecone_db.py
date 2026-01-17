import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "ipl-stats-demo"

# Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,  # text-embedding-3-large
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
vector_store = PineconeVectorStore(
    index=index,                    
    embedding=embedding,
    namespace="langchain-examples"
)

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )
docs = [doc1, doc2, doc3, doc4, doc5]

#vector_store.add_documents(docs)
# docs = vector_store.similarity_search(
#     query="IPL players",
#     k=10
# )

# for d in docs:
#     print("CONTENT:", d.page_content)
#     print("METADATA:", d.metadata)
results = vector_store.similarity_search(
    query="Which among them are bowler?",
    k=2
)

# for doc in results:
#     print("CONTENT:", doc.page_content)
#     print("METADATA:", doc.metadata)
#     print("-" * 40)

# meta-data filtering
results = vector_store.similarity_search_with_score(
    query="Chennai Super Kings players",
    filter={"team": "Chennai Super Kings"},
    k=5
)

for doc, score in results:
    print(f"SCORE: {score:.4f}")
    print("CONTENT:", doc.page_content)
    print("METADATA:", doc.metadata)
    print("-" * 40)
