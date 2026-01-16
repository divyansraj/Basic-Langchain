import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()   

llm = ChatOpenAI(
    model="gpt-4.1", # Or "claude-3-5-sonnet", "gemini-1.5-pro"
    openai_api_key=os.getenv("GITHUB_TOKEN"),
    openai_api_base="https://models.inference.ai.azure.com"
)
messages= [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello, who won the world series in 2020?"),
]
result = llm.invoke(messages)
print("AI: ", result.content)
messages.append(AIMessage(content=result.content))
print("Current conversation:", messages)