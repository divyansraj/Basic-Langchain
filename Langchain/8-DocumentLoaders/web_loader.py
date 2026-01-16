import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv
load_dotenv()

os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; LangChainBot/1.0)"

llm = ChatOpenAI(model="gpt-4.1-nano")
template1 = PromptTemplate(
    template="Answer this question {question} from the following content provided. \n {web_text}",
    input_variables=["question","web_text"]
)
loader = WebBaseLoader("https://www.ibm.com/think/insights/artificial-intelligence-future")
docs = loader.load()
parser  =StrOutputParser()

chain = template1 | llm | parser

result = chain.invoke({
    "question": "Will aI replace software developers in future?",
    "web_text": docs[0].page_content
})

print("Answer:\n", result)