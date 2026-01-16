from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-nano")

template1 = PromptTemplate(
    template="Write a detailed notes with comapring this to Java springboot. \n {text}",
    input_variables=["text"]
)
template2 =PromptTemplate(
    template="Write a summary about this following document. \n {doc}",
    input_variables=["doc"]
)

parser  =StrOutputParser()

loader1 = TextLoader("C:/Users/divyankk/Downloads/GenAI/LLMs/8-DocumentLoaders/z1.txt", encoding="utf8")
loader2 = Docx2txtLoader("C:/Users/divyankk/Downloads/GenAI/LLMs/8-DocumentLoaders/z2.docx")

text_docs = loader1.load()
docx_docs = loader2.load()

text_content = text_docs[0].page_content
docx_content = docx_docs[0].page_content


parallel_chain = RunnableParallel({
    "first": RunnableLambda(lambda x: {"text": x["text"]}) | template1 | llm | parser,
    "second": RunnableLambda(lambda x: {"doc": x["doc"]}) | template2 | llm | parser
})
result = parallel_chain.invoke({
    "text": text_content,
    "doc": docx_content
})

print("Detailed Notes:\n", result["first"])
print("\nSummary:\n", result["second"])
