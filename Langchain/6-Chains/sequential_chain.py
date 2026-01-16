from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-nano"
)

template1 =  PromptTemplate(
    template="Give a detailed explanation on {topic}",
    input_variables=["topic"]
)

template2= PromptTemplate(
    template="Summarize the following detailed explanation into 5 points: {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = template1 | llm | parser| template2 | llm | parser

result = chain.invoke({"topic": "Artificial Intelligence" })

print(result)