from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-nano")

template1 = PromptTemplate(
    template="Extract the key points from the following PDF content. \n {pdf_text}",
    input_variables=["pdf_text"]
)
parser  =StrOutputParser()

loader = PyMuPDFLoader("C:/Users/divyankk/Downloads/GenAI/LLMs/8-DocumentLoaders/z4.pdf")
doc = loader.load()

chain = template1 | llm | parser

result = chain.invoke({"pdf_text": doc[0].page_content})

print("Key Points from PDF:\n", result)
