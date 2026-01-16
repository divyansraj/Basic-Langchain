from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

loader= DirectoryLoader(
    path = "C:/Users/divyankk/Downloads/books",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader
)

docs = loader.load()
print(docs[0].page_content)
print(docs[0].metadata)

llm = ChatOpenAI(model="gpt-4.1-nano")
template1 = PromptTemplate(
    template="Extract the key points from the following PDF content. \n {pdf_text}",
    input_variables=["pdf_text"]
)