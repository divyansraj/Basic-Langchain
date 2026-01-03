import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1", # Or "claude-3-5-sonnet", "gemini-1.5-pro"
    openai_api_key=os.getenv("GITHUB_TOKEN"),
    openai_api_base="https://models.inference.ai.azure.com"
)
template= ChatPromptTemplate([
    ('system','You are a helpful {domain} expert.'),
    ('human','Explain in simple terms what is {user_input}.')
])

prompt =template.invoke({
    'domain': 'cricket',
    'user_input': 'lbw'
})
print(prompt)
result = llm.invoke(prompt)
print("AI: ", result.content)