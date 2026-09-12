from dotenv import load_dotenv
load_dotenv()
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch

prompt1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Summarize the following text \n {text}",
    input_variables = ['text']
)

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

parser = StrOutputParser()

report_chain = RunnableSequence(
    prompt1,
    llm,
    parser
)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 300, RunnableSequence(
        prompt2, llm, parser
    )),
    RunnablePassthrough()
)

final_chain = RunnableSequence(
    report_chain,
    branch_chain
)

print(final_chain.invoke(
    {'topic' : 'Ruusia vs Ukraine'}
))