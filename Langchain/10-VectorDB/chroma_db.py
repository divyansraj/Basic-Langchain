import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='my_chroma_db',
    collection_name='sample'
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
ids = [
    "kohli",
    "rohit",
    "dhoni",
    "bumrah",
    "jadeja",
]

# add documents
# vector_store.add_documents(docs, ids=ids)

# view documents

# data = vector_store.similarity_search(
#     query="players",
#     k=10
# )
# for d in data:
#     print("CONTENT:", d.page_content)
#     print("METADATA:", d.metadata)
#     print("-" * 40)

# results = vector_store.similarity_search_with_score(
#     query="",
#     filter={"team": "Chennai Super Kings"},
#     k=2
# )

# for doc, score in results:
#     print(f"SCORE: {score:.4f}")
#     print("CONTENT:", doc.page_content)
#     print("METADATA:", doc.metadata)
#     print("-" * 40)


#update docs
updated_doc = Document(
    page_content="Virat Kohli is an elite batsman and former India captain.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.add_documents(
    documents=[updated_doc],
    ids=["kohli"]  
)
data = vector_store.similarity_search(
    query="elite batsman",
    k=1
)
for d in data:
    print("CONTENT:", d.page_content)
    print("METADATA:", d.metadata)
    print("-" * 40)

# delete docs

#vector_store.delete(ids=["kohli"])

# view
data = vector_store.similarity_search(
    query="elite batsman virat kohli",
    k=1
)
for d in data:
    print("CONTENT:", d.page_content)
    print("METADATA:", d.metadata)
    print("-" * 40)