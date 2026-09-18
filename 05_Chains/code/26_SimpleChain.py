from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

prompt = PromptTemplate(
    template = "Generate 5 interesting facts about this {topic} in short",
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke(
    {'topic' : 'Basketball'}
)

print(result)