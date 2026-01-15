from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("C:/Users/divyankk/Downloads/GenAI/LLMs/9-TextSplitting/z1.txt", encoding="utf-8")
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    separator=""
)
result = splitter.split_documents(docs)

print(result[0].page_content)