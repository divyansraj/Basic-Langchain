from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

llm1= ChatOpenAI(model="gpt-4.1-nano")
llm2= ChatOpenAI(model="grok-3")

template1= PromptTemplate(
    template="Tell me a funny understandable joke about topic {topic}",
    input_variables=["topic"]
)
template2= PromptTemplate(
    template="Explain the following joke: \n {topic}",
    input_variables=["topic"]
)
parser = StrOutputParser()

joke = RunnableSequence(template1,llm1,parser)

joke_with_explanation = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(template2,llm1,parser)
})
final_chain = RunnableSequence(joke,joke_with_explanation)

result = final_chain.invoke({"topic": "AI"})
print(result["joke"])
print(result["explanation"])