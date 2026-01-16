import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template="Summarize in 5 of the following text: {text}",
    input_variables=["text"]
)   

parser = StrOutputParser(); 
#Main Advantage we can make use of chains without extracting .content and directly call parser
chain = template1 | model | parser | template2 | model | parser

prompt = chain.invoke({"topic": "The impact of climate change on global agriculture"})  

print(prompt)