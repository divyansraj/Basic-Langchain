from langchain_community.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4.1")
template1 = PromptTemplate(
    template="Is there any customer named David? \n {csv_data}",
    input_variables=["csv_data"]
)   
parser  =StrOutputParser()

loader = CSVLoader("C:/Users/divyankk/Downloads/GenAI/LLMs/8-DocumentLoaders/customers-data.csv")
docs = loader.load()    
csv_text = "\n".join(doc.page_content for doc in docs)

chain = template1 | llm | parser

result = chain.invoke({"csv_data": csv_text})
print("Insights from CSV Data:\n", result)