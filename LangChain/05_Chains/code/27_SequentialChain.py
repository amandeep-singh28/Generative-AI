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

prompt1 = PromptTemplate(
    template = "Generate information about this {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Generate 5 small points from the following text \n {text}",
    input_variables = ['text']
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke(
    {'topic' : 'Cricket'}
)

print(result)

chain.get_graph().print_ascii()