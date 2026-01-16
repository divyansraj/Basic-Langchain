import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  
from langchain_core.prompts import PromptTemplate   
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(ge=20,description="The age of the person")
    occupation: str = Field(description="The occupation of the person")
    birthplace: str = Field(description="The birthplace of the person")

parser = PydanticOutputParser(pydantic_object=Person)
#Structured output with validation

template= PromptTemplate(
    template="Provide details about a person from field {field} including their name, age, occupation, and birthplace. \n {format_instruction}",
    input_variables=["field"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"field": "head of oneplus mobile company"})

print(result)


