from dotenv import load_dotenv
load_dotenv()
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

prompt1 = PromptTemplate(
    template = "Generate a tweet about {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Generate a LinkedIn post about {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, llm, parser),
    'LinkedIn' : RunnableSequence(prompt2, llm, parser)
})

result = parallel_chain.invoke(
    {'topic' : 'Machine learning'}
)

print(result)