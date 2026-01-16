from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

llm=ChatOpenAI(model="gpt-4.1-nano")

class Classify(BaseModel):
    sentiment: Literal["positive","negative"] = Field(description="The sentiment of the review weather it is positive or negative")

pydanticParser = PydanticOutputParser(pydantic_object=Classify)

template1 = PromptTemplate(
    template="Classify the sentiment of the review into positive or negative. \n {text} \n {format_instruction}",
    input_variables=["text"],
    partial_variables={"format_instruction": pydanticParser.get_format_instructions()}
)

template2 = PromptTemplate(
    template="Give a suitable response for the positive sentiment. \n {feedback}",
    input_variables=["feedback"]
)
template3 = PromptTemplate(
    template="Give a suitable response for the negative sentiment. \n {feedback}",
    input_variables=["feedback"]
)

strParser = StrOutputParser()

classifier_chain = template1 | llm | pydanticParser

branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive', template2 | llm | strParser),
    (lambda x:x.sentiment=='negative', template3 | llm | strParser),
    RunnableLambda(lambda x: "Sentiment cannot be detected")
)
conditional_chain = classifier_chain | branch_chain
result = conditional_chain.invoke({"text": "The students are playing in the cricket ground."})

print("Classifier Chain Output:",result )
