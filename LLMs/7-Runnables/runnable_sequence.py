from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm1 = ChatOpenAI(model="gpt-4.1-nano")
llm2 = ChatOpenAI(model="grok-3")

template1= PromptTemplate(
    template="What is the capital of {country} and then its population?",
    input_variables=["country"]
)
template2= PromptTemplate(
    template="Tell me which model are you and developed by which company? Write a brief history and culture of this country. \n {text}",
    input_variables=["text"]
)
parser = StrOutputParser()
sequence = RunnableSequence(template1,llm1,parser,template2,llm2,parser)

result = sequence.invoke({"country": "India"})

print(result)

