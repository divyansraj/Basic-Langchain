import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1", # Or "claude-3-5-sonnet", "gemini-1.5-pro"
    openai_api_key=os.getenv("GITHUB_TOKEN"),
    openai_api_base="https://models.inference.ai.azure.com"
)

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])
chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
print(chat_history)

chat = chat_template.invoke({
    'chat_history': chat_history,
    'input': 'I havent got refund yet.'
})
print(chat)

result = llm.invoke(chat)
print("AI: ", result.content)
