from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate   
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
from dotenv import load_dotenv
load_dotenv()

llm1= ChatOpenAI(model="gpt-4.1-nano")
llm2= ChatOpenAI(model="grok-3")
template1= PromptTemplate(
   template="Tell me which model are you and developed by which company? Write a brief note on future of {topic}",
    input_variables=["topic"]
)
template2= PromptTemplate(
    template="Tell me which model are you and developed by which company? Write a brief note on future of {topic}",
    input_variables=["topic"]
)
parser = StrOutputParser()

parallel = RunnableParallel({
    "chatGpt": RunnableSequence(template1,llm1,parser),
    "grok": RunnableSequence(template2,llm2,parser)
})

result = parallel.invoke({"topic": "jobs in Software field in 2026"})
print(result)
