import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()   

llm = ChatOpenAI(
    model="gpt-4.1", # Or "claude-3-5-sonnet", "gemini-1.5-pro"
    openai_api_key=os.getenv("GITHUB_TOKEN"),
    openai_api_base="https://models.inference.ai.azure.com"
)

chat_history = []
while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input.lower() in ["exit","quit"]:
        print(chat_history )
        break
    result= llm.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: ", result.content)
    