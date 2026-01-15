from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("C:/Users/divyankk/Downloads/GenAI/LLMs/9-TextSplitting/z4.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0
)

chunks = splitter.split_documents(docs)
print(len(chunks))
print(chunks)