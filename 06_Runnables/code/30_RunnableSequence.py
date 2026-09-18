from dotenv import load_dotenv
load_dotenv()
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = "Explain the following joke {text}",
    input_variables = ['text']
)

parser = StrOutputParser()

chain = RunnableSequence(
    prompt1,
    llm,
    parser,
    prompt2,
    llm,
    parser
)

result = chain.invoke(
    {"topic" : "AI"}
)

print(result)