# from langchain_openai import OpenAI
# from dotenv import load_dotenv
# load_dotenv()

# llm = OpenAI(model='gpt-3.5-turbo-instruct')
# response = llm.invoke("Tell me a joke about programming.")
# print(response)

# import os
# from dotenv import load_dotenv
# from langchain_perplexity import ChatPerplexity

# load_dotenv()
# llm = ChatPerplexity(model="sonar", temperature=0)

# response = llm.invoke("What is the latest news about LangChain in late 2025?")
# print(response.content)


# --- CHAT MODEL --- #
import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

load_dotenv()

# GitHub Models uses the OpenAI-compatible endpoint
llm = ChatOpenAI(
    model="o3-mini", # Or "claude-3-5-sonnet", "gemini-1.5-pro"
    openai_api_key=os.getenv("GITHUB_TOKEN"),
    openai_api_base="https://models.inference.ai.azure.com"
)

response = llm.invoke("What is the capital of India?")
print(response)

