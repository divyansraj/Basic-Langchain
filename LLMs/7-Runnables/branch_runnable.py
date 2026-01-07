from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv
load_dotenv()

llm1= ChatOpenAI(model="gpt-4.1-nano")
llm2= ChatOpenAI(model="grok-3")

template1= PromptTemplate(
    template="Make a report on the following topic {topic} and then tell me how many words your report is.",
    input_variables=["topic"]
)
template2= PromptTemplate(
    template="Summarize the following report in 200 words: \n {topic} \n and then tell me how many words your report is.",
    input_variables=["topic"]
)
parser = StrOutputParser()

report = RunnableSequence(template1,llm1,parser)
summary = RunnableBranch(
    (lambda z: len(z.split())> 500, RunnableSequence(template2,llm2,parser)),
    (lambda z: len(z.split())<= 500, RunnablePassthrough()),
    RunnableLambda(lambda x: "Report length could not be determined.")
)

final_chain =RunnableSequence(report,summary)
result = final_chain.invoke({"topic": "Climate Change and its impact on global economy in 1000"})
print(result)
