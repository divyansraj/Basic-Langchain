from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Summarize in 5 of the following text: {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "The impact of climate change on global agriculture"}  )

result1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result1})

result2 = model.invoke(prompt2)

print(result2.content)